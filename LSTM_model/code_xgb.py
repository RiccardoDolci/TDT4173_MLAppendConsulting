#!/usr/bin/env python
"""
XGBoost quantile model with Optuna tuning.

- Loads training_dataset.csv (features + target).
- Splits chronologically: last 150 rows per rm_id -> validation.
- Tunes XGBoost quantile hyperparameters with Optuna (pinball loss, alpha=0.2).
- Retrains on train+val at best iteration.
- Predicts on test features; applies per-rm_id cummax and writes submission.csv.

Requires:
    xgboost >= 2.0.0
    optuna >= 3.x
"""

import pathlib
import os
import json
import pandas as pd
import numpy as np

BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Toggle hyperparameter tuning
ENABLE_HYPERPARAM_TUNING = True
HYPEROPT_N_TRIALS = 60
HYPEROPT_TIMEOUT = None  # seconds or None

try:
    import xgboost as xgb
except ImportError as e:
    raise RuntimeError("Missing dependency: install xgboost>=2.0.0") from e

def _encode_rm_id_as_codes(X_train_full: pd.DataFrame,
                           X_val: pd.DataFrame,
                           X_test: pd.DataFrame):
    """
    Make rm_id categorical with training categories, then map to stable int codes for XGBoost.
    """
    if 'rm_id' not in X_train_full.columns:
        return X_train_full, X_val, X_test, []

    X_train_full = X_train_full.copy()
    X_train_full['rm_id'] = X_train_full['rm_id'].astype('category')
    rm_categories = X_train_full['rm_id'].cat.categories.tolist()

    def _align(df):
        df = df.copy()
        df['rm_id'] = pd.Categorical(df['rm_id'], categories=rm_categories)
        df['rm_id'] = df['rm_id'].cat.codes.astype('int32')
        return df

    X_val = _align(X_val)
    X_test = _align(X_test)
    X_train_full['rm_id'] = X_train_full['rm_id'].cat.codes.astype('int32')

    return X_train_full, X_val, X_test, rm_categories


def main():
    # Import data preparation helpers from the module
    import data_preparation

    training_file = BASE_DIR / 'data' / 'training_dataset.csv'
    if not training_file.exists():
        print(f"Training dataset not found at {training_file}. Running data_preparation.main() to create it...")
        data_preparation.main()

    # Load training data and extract exported metadata (dates, recency weights) if present
    training_df = pd.read_csv(training_file, index_col=0)

    sample_weight_series = None
    if 'recency_weight' in training_df.columns:
        sample_weight_series = training_df['recency_weight']

    # drop meta columns from features
    drop_meta = [c for c in ['forecast_start_date', 'forecast_end_date', 'recency_weight'] if c in training_df.columns]
    X_train = training_df.drop(columns=['target'] + drop_meta)
    y_train = training_df['target']

    # Build test set
    receivals, purchase_orders, materials, prediction_mapping = data_preparation.load_data()
    daily_receivals, daily_po = data_preparation.aggregate_daily_data(receivals, purchase_orders, materials)
    test_df_raw = prediction_mapping.set_index('ID')
    X_test = data_preparation.create_features(test_df_raw, daily_receivals, daily_po)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Chronological holdout: last 150 rows per rm_id -> validation
    print("Creating chronological holdout: last 150 rows per rm_id -> validation")
    val_idx = X_train.groupby('rm_id').tail(150).index
    X_val = X_train.loc[val_idx].copy()
    y_val = y_train.loc[val_idx].copy()
    train_idx = X_train.index.difference(val_idx)
    X_train_full = X_train.loc[train_idx].copy()
    y_train_full = y_train.loc[train_idx].copy()
    print(f"Training rows: {len(X_train_full)}, Validation rows: {len(X_val)}")

    # Encode rm_id consistently for XGBoost
    X_train_full, X_val, X_test, rm_categories = _encode_rm_id_as_codes(X_train_full, X_val, X_test)

    # Prepare sample weights aligned to the training index (default 1.0)
    if sample_weight_series is None:
        sample_weight_series = pd.Series(1.0, index=X_train.index)
    else:
        sample_weight_series = sample_weight_series.reindex(X_train.index).fillna(1.0)

    sample_weight_val = sample_weight_series.loc[val_idx].values
    sample_weight_train = sample_weight_series.loc[train_idx].values

    # Create QuantileDMatrix datasets with sample weights so XGBoost uses recency weighting
    dtrain = xgb.QuantileDMatrix(X_train_full, y_train_full, weight=sample_weight_train)
    dvalid = xgb.QuantileDMatrix(X_val, y_val, weight=sample_weight_val, ref=dtrain)
    dtest  = xgb.QuantileDMatrix(X_test, ref=dtrain)

    # Base params (fixed for both tuning and final fit)
    base_params = {
        "objective": "reg:quantileerror",  # native pinball loss
        "quantile_alpha": 0.2,
        "tree_method": "hist",             # required for QuantileDMatrix
        "eval_metric": "quantile",         # evaluates pinball loss
        "seed": 42,
        "verbosity": 0,
    }

    best_params = {}
    best_iter = None

    if ENABLE_HYPERPARAM_TUNING:
        try:
            import optuna
            from optuna.samplers import TPESampler
        except Exception as e:
            raise RuntimeError("ENABLE_HYPERPARAM_TUNING=True but optuna is not installed.") from e

        print(f"Running Optuna tuning (trials={HYPEROPT_N_TRIALS})...")

        def objective(trial: "optuna.trial.Trial"):
            params = base_params.copy()
            params.update({
                "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 64.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 10.0),
                # Optional: max_bin tuning if needed
                # "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
            })

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=20000,
                evals=[(dtrain, "train"), (dvalid, "valid")],  # valid last to drive early stopping
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            # Store iteration for later reuse
            trial.set_user_attr("best_iteration", booster.best_iteration)
            trial.set_user_attr("best_score", booster.best_score)
            return float(booster.best_score)

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=HYPEROPT_N_TRIALS, timeout=HYPEROPT_TIMEOUT)

        best_params = study.best_params
        best_iter = int(study.best_trial.user_attrs.get("best_iteration", 1000))
        print(f"Best params: {best_params}")
        print(f"Best iteration: {best_iter}")
    else:
        # Reasonable defaults if tuning disabled
        best_params = {
            "eta": 0.05,
            "max_depth": 6,
            "min_child_weight": 1.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
        # Determine best_iter via a quick early-stopped fit
        params = base_params.copy()
        params.update(best_params)
        print("Estimating best iteration with early stopping...")
        tmp_booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=20000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        best_iter = tmp_booster.best_iteration

    # Final training on train+val using tuned params and best iteration
    print("Retraining final model on train+val at best iteration...")
    X_combined = pd.concat([X_train_full, X_val], axis=0)
    y_combined = pd.concat([y_train_full, y_val], axis=0)
    # combine sample weights for train+val
    sw_combined = pd.concat([sample_weight_series.loc[X_train_full.index], sample_weight_series.loc[X_val.index]], axis=0).reindex(X_combined.index).fillna(1.0).values
    dcomb = xgb.QuantileDMatrix(X_combined, y_combined, weight=sw_combined)
    final_params = base_params.copy()
    final_params.update(best_params)
    booster_final = xgb.train(
        params=final_params,
        dtrain=dcomb,
        num_boost_round=(best_iter + 1 if best_iter is not None else 1000),
        verbose_eval=False,
    )

    # Save model and metadata
    model_out_dir = BASE_DIR / 'outputs' / 'model'
    os.makedirs(model_out_dir, exist_ok=True)
    model_file = model_out_dir / 'best_xgb_model.json'
    meta_file = model_out_dir / 'best_xgb_model_meta.json'
    booster_final.save_model(str(model_file))
    meta = {
        'feature_columns': X_train.columns.tolist(),
        'categorical_rm_id_categories': rm_categories,
        'quantile_alpha': 0.2,
        'library': 'xgboost',
        'best_iteration': int(best_iter) if best_iter is not None else None,
        'tuning_used': bool(ENABLE_HYPERPARAM_TUNING),
        'best_params': best_params,
    }
    with open(meta_file, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved model to {model_file} and metadata to {meta_file}")

    # Predict
    print("Generating predictions...")
    preds = booster_final.predict(dtest)
    preds = np.asarray(preds, dtype=float)
    preds[preds < 0] = 0.0

    # Submission with per-rm_id cummax
    submission = pd.DataFrame({'ID': X_test.index, 'predicted_weight': preds})
    mapping_small = prediction_mapping[['ID', 'rm_id', 'forecast_end_date']].copy()
    mapping_small['forecast_end_date'] = pd.to_datetime(mapping_small['forecast_end_date'])
    merged_sub = submission.merge(mapping_small, on='ID', how='left').sort_values(['rm_id', 'forecast_end_date'])
    merged_sub['predicted_weight'] = merged_sub.groupby('rm_id')['predicted_weight'].cummax()
    submission = merged_sub.sort_values('ID')[['ID', 'predicted_weight']].reset_index(drop=True)

    submission_filename = BASE_DIR / 'data' / 'submission_xgb.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"Saved submission to {submission_filename}")

if __name__ == '__main__':
    main()
