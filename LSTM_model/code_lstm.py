#!/usr/bin/env python

"""LSTM_model training & prediction runner (single 150-day holdout + full-data refit).

Responsibilities
- Load prepared training dataset (data/training_dataset.csv).
- Train LightGBM quantile regressor (alpha=0.2) with early stopping on last-150/day per rm_id holdout.
- Save the validation-fitted model and metadata.
- Refit on full dataset (train + validation) using best params and best_iteration_.
- Build test features from data/prediction_mapping.csv, predict, enforce per-rm_id monotonicity, save submission.

Run
    python LSTM_model/code_lstm.py
"""

import pathlib
import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Toggle hyperparameter tuning (Optuna). Default: False
ENABLE_HYPERPARAM_TUNING = False
HYPEROPT_N_TRIALS = 60
HYPEROPT_TIMEOUT = None  # seconds, or None

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
except ImportError as e:
    raise RuntimeError("Missing dependencies: install lightgbm and scikit-learn") from e


def main():
    # Import data preparation helpers
    import data_preparation

    training_file = BASE_DIR / 'data' / 'training_dataset.csv'
    if not training_file.exists():
        print(f"Training dataset not found at {training_file}. Running data_preparation.main() to create it...")
        data_preparation.main()

    training_df = pd.read_csv(training_file, index_col=0)

    # Extract optional sample weights
    sample_weight_series = None
    if 'recency_weight' in training_df.columns:
        sample_weight_series = training_df['recency_weight']

    # Drop meta columns from features
    drop_meta = [c for c in ['forecast_start_date', 'forecast_end_date', 'recency_weight'] if c in training_df.columns]
    X_all = training_df.drop(columns=['target'] + drop_meta)
    y_all = training_df['target']

    # Build test set
    receivals, purchase_orders, materials, prediction_mapping = data_preparation.load_data()
    daily_receivals, daily_po = data_preparation.aggregate_daily_data(receivals, purchase_orders, materials)
    test_df_raw = prediction_mapping.set_index('ID')
    X_test = data_preparation.create_features(test_df_raw, daily_receivals, daily_po)
    # Align test columns to training
    X_test = X_test.reindex(columns=X_all.columns, fill_value=0)

    # Single chronological holdout: last 150 rows per rm_id -> validation
    print("Creating chronological holdout: last 150 rows per rm_id -> validation")
    val_idx = X_all.groupby('rm_id').tail(150).index
    X_val = X_all.loc[val_idx].copy()
    y_val = y_all.loc[val_idx].copy()
    train_idx = X_all.index.difference(val_idx)
    X_train = X_all.loc[train_idx].copy()
    y_train = y_all.loc[train_idx].copy()
    print(f"Training rows: {len(X_train)}, Validation rows: {len(X_val)}")

    # Sample weights
    if sample_weight_series is None:
        sample_weight_series = pd.Series(1.0, index=X_all.index)
    else:
        sample_weight_series = sample_weight_series.reindex(X_all.index).fillna(1.0)

    sw_tr = sample_weight_series.loc[train_idx].values
    sw_va = sample_weight_series.loc[val_idx].values

    # Categorical handling
    if 'rm_id' in X_train.columns:
        X_train['rm_id'] = X_train['rm_id'].astype('category')
        X_val['rm_id'] = pd.Categorical(X_val['rm_id'], categories=X_train['rm_id'].cat.categories)
        X_test['rm_id'] = pd.Categorical(X_test['rm_id'], categories=X_train['rm_id'].cat.categories)

    # Base params (quantile 0.2)
    base_params = dict(
        objective='quantile',
        alpha=0.2,
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        n_jobs=-1,
    )

    # Train model with early stopping on validation
    if ENABLE_HYPERPARAM_TUNING:
        try:
            import optuna
            from optuna.integration import LightGBMPruningCallback
        except Exception as e:
            raise RuntimeError("ENABLE_HYPERPARAM_TUNING is True but Optuna is not installed.") from e

        print(f"Running hyperparameter tuning with Optuna (trials={HYPEROPT_N_TRIALS})...")

        def quantile_loss(y_true, y_pred, q=base_params.get('alpha', 0.2)):
            e = y_true - y_pred
            return (np.maximum(q * e, (q - 1) * e)).mean()

        def objective(trial):
            if y_train.nunique() <= 1 or np.nanstd(y_train) == 0:
                return 1e12

            zero_var_cols = X_train.columns[X_train.nunique() <= 1].tolist()
            if len(zero_var_cols) > 0:
                X_tr_trial = X_train.drop(columns=zero_var_cols)
                X_va_trial = X_val.drop(columns=[c for c in zero_var_cols if c in X_val.columns])
            else:
                X_tr_trial = X_train
                X_va_trial = X_val

            params = {
                'objective': 'quantile',
                'alpha': base_params.get('alpha', 0.2),
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'n_jobs': -1,
            }

            model_t = LGBMRegressor(**params)
            pruning_cb = LightGBMPruningCallback(trial, 'quantile')
            try:
                y_va_trial = y_val.loc[X_va_trial.index]
                model_t.fit(
                    X_tr_trial, y_train.loc[X_tr_trial.index],
                    sample_weight=sample_weight_series.loc[X_tr_trial.index].values,
                    eval_set=[(X_va_trial, y_va_trial)],
                    eval_sample_weight=[sample_weight_series.loc[X_va_trial.index].values],
                    eval_metric='quantile',
                    callbacks=[lgb.early_stopping(100, verbose=False), pruning_cb],
                )
            except Exception:
                return 1e12

            val_preds = model_t.predict(X_va_trial)
            return float(quantile_loss(y_val.values, val_preds, q=params['alpha']))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=HYPEROPT_N_TRIALS, timeout=HYPEROPT_TIMEOUT)

        best_params = study.best_params
        print(f"Best params: {best_params}")

        final_params = best_params.copy()
        final_params.update({'objective': 'quantile', 'alpha': base_params.get('alpha', 0.2), 'random_state': 42, 'n_jobs': -1})

        model = LGBMRegressor(**final_params)
        print("Fitting tuned model on train+val (to finalize hyperparameters for refit)...")
        # Fit on train for early stopping bookkeeping (optional)
        model.fit(
            X_train, y_train,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[sw_va],
            eval_metric='quantile',
            callbacks=[lgb.early_stopping(100, verbose=True)],
        )
        best_params_meta = study.best_params
    else:
        model = LGBMRegressor(**base_params)
        print("Fitting model with early stopping...")
        model.fit(
            X_train, y_train,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[sw_va],
            eval_metric='quantile',
            callbacks=[lgb.early_stopping(100, verbose=True)],
        )
        best_params_meta = {}

    # Save validation-fitted model and metadata
    model_out_dir = BASE_DIR / 'outputs' / 'model'
    os.makedirs(model_out_dir, exist_ok=True)
    model_file = model_out_dir / 'best_lgbm_model.joblib'
    meta_file = model_out_dir / 'best_lgbm_model_meta.json'
    joblib.dump(model, model_file)
    meta = {
        'feature_columns': X_all.columns.tolist(),
        'categorical_rm_id_categories': X_train['rm_id'].cat.categories.tolist() if 'rm_id' in X_train.columns else [],
        'alpha': base_params.get('alpha', 0.2),
        'tuning_used': bool(ENABLE_HYPERPARAM_TUNING),
        'best_params': best_params_meta if 'best_params_meta' in locals() else {},
        'best_iteration_': int(getattr(model, 'best_iteration_', 0) or 0),
    }
    with open(meta_file, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved model to {model_file} and metadata to {meta_file}")

    # --- Refit on full dataset using best settings and best_iteration_ ---
    print("Refitting final model on full dataset (train + validation) using best settings...")
    with open(meta_file, 'r', encoding='utf-8') as fh:
        meta_loaded = json.load(fh)

    # Determine params for refit
    if meta_loaded.get('best_params'):
        refit_params = meta_loaded['best_params'].copy()
        refit_params.update({
            'objective': 'quantile',
            'alpha': meta_loaded.get('alpha', 0.2),
            'random_state': 42,
            'n_jobs': -1,
        })
    else:
        refit_params = model.get_params()

    # Use best_iteration_ as n_estimators if present
    best_it = meta_loaded.get('best_iteration_', 0)
    if best_it and int(best_it) > 0:
        refit_params['n_estimators'] = int(best_it)

    # Combine full dataset
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)
    sw_combined = pd.concat(
        [pd.Series(sw_tr, index=X_train.index),
         pd.Series(sw_va, index=X_val.index)],
        axis=0
    ).reindex(X_combined.index).fillna(1.0).values

    # Categorical alignment on combined data
    if 'rm_id' in X_combined.columns:
        X_combined['rm_id'] = X_combined['rm_id'].astype('category')
        X_test['rm_id'] = pd.Categorical(X_test['rm_id'], categories=X_combined['rm_id'].cat.categories)

    refit_model = LGBMRegressor(**refit_params)
    refit_model.fit(X_combined, y_combined, sample_weight=sw_combined)

    # Save refit model and metadata
    model_file_full = model_out_dir / 'best_lgbm_model_full.joblib'
    meta_file_full = model_out_dir / 'best_lgbm_model_full_meta.json'
    joblib.dump(refit_model, model_file_full)
    meta_loaded.update({
        'refit_on_full_data': True,
        'best_iteration_used': int(best_it) if best_it else None,
        'final_feature_columns': X_combined.columns.tolist(),
    })
    with open(meta_file_full, 'w', encoding='utf-8') as fh:
        json.dump(meta_loaded, fh, indent=2)
    print(f"Saved full-data refit model to {model_file_full} and metadata to {meta_file_full}")

    # Predict with refit model
    preds = refit_model.predict(X_test)
    preds[preds < 0] = 0
    submission = pd.DataFrame({'ID': X_test.index, 'predicted_weight': preds})
    mapping_small = prediction_mapping[['ID', 'rm_id', 'forecast_end_date']].copy()
    mapping_small['forecast_end_date'] = pd.to_datetime(mapping_small['forecast_end_date'])
    merged_sub = submission.merge(mapping_small, on='ID', how='left')
    merged_sub = merged_sub.sort_values(['rm_id', 'forecast_end_date'])
    merged_sub['predicted_weight'] = merged_sub.groupby('rm_id')['predicted_weight'].cummax()
    submission = merged_sub.sort_values('ID')[['ID', 'predicted_weight']].reset_index(drop=True)
    submission_filename = BASE_DIR / 'data' / 'submission_lgbm.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"Saved submission to {submission_filename}")


if __name__ == '__main__':
    main()
