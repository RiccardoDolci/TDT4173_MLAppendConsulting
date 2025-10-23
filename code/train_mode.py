import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from pathlib import Path
import optuna

# --- Load the dataset
# Read the data into pandas 
df = pd.read_csv('data/mod_data/merged_clean_data.csv')
print("Successfully loaded data from Azure.")
df['date_arrival'] = pd.to_datetime(df['date_arrival'], utc=True)
# --- 2. Data Type Conversion ---
print("Converting data types...")
# Ensure categorical columns are treated as such by the model
categorical_cols = ['rm_id', 'day', 'month', 'day_of_week', 'week_of_year','is_closure_day', 'first_day_of_year']
for col in categorical_cols:
    df[col] = df[col].astype('category')


# --- 3. Time-Based Split ---
print("Splitting data into training (before 2024) and validation (2024) sets...")
train_df = df[df['date_arrival'] < '2024-01-01'].copy()
val_df = df[df['date_arrival'] >= '2024-01-01'].copy()

print(f"Training set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")


# --- 4. Define Features (X) and Target (y) ---
# List of columns to use as features for the model
# NOTE: We remove 'date_arrival' as it's not a direct feature for the tree model
features_to_use = [
    'rm_id', 'month', 'day', 'day_of_week', 'week_of_year','is_closure_day',
    'first_day_of_year',
    'lag_1_day', 'lag_7_days', 'lag_14_days', 'lag_28_days',
    'rolling_mean_7_days', 'rolling_mean_14_days', 'rolling_mean_28_days',
    'rolling_std_7_days', 'rolling_std_14_days', 'rolling_std_28_days'
]
target_col = 'cum_net_weight'

X_train = train_df[features_to_use]
y_train = train_df[target_col]
X_val = val_df[features_to_use]
y_val = val_df[target_col]

print("\nData successfully split and prepared for modeling.")

# --- 5. Custom Evaluation Metric ---
def quantile_error_0_2(y_true, y_pred):
    y_true = np.array(y_true)
    loss = np.maximum(0.2 * (y_true - y_pred), 0.8 * (y_pred - y_true))
    return np.mean(loss)

# --- 6. Optuna Objective Function ---
def objective(trial):
    """
    This is the function Optuna will try to minimize.
    It takes a 'trial' object, suggests hyperparameters, trains a model,
    and returns the validation score.
    """
    # Define the search space for hyperparameters
    params = {
        'objective': 'quantile',
        'alpha': 0.2,
        'metric': 'quantile',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1  # Suppress verbose output during trials
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='quantile',
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=categorical_cols
    )

    preds = model.predict(X_val)
    # The custom metric function is not needed here as LGBM's quantile metric is the same.
    # We can get the score directly from the model's best_score_ attribute.
    score = model.best_score_['valid_0']['quantile']
    
    return score

# --- 7. Run the Hyperparameter Tuning Study ---
print("\n--- Starting Hyperparameter Tuning with Optuna ---")
# Create a study object and specify the direction to 'minimize' the objective
study = optuna.create_study(direction='minimize')

# Start the optimization process
study.optimize(objective, n_trials=50) # Run 50 different trials

print("\n--- Hyperparameter Tuning Finished ---")
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (Quantile Error): {best_trial.value:,.4f}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# --- 8. Train the Final Model with Best Hyperparameters ---
print("\n--- Training Final Model with Best Hyperparameters ---")
best_params = best_trial.params
# Add back the fixed parameters
best_params['objective'] = 'quantile'
best_params['alpha'] = 0.2
best_params['metric'] = 'quantile'
best_params['n_estimators'] = 2000 # Use a higher number and early stopping
best_params['n_jobs'] = -1
best_params['random_state'] = 42

final_model = lgb.LGBMRegressor(**best_params)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='quantile',
    callbacks=[lgb.early_stopping(100, verbose=True)],
    categorical_feature=categorical_cols
)

# --- 9. Evaluate and Save ---
# Predict on the validation set (predictions are on the log scale)
val_predictions = final_model.predict(X_val)


# Ensure predictions are not negative
val_predictions[val_predictions < 0] = 0

# Now, evaluate the error on the ORIGINAL scale
# We compare the original y_val with the inverse-transformed predictions
final_error = quantile_error_0_2(y_val, val_predictions)

print(f"\nFinal Validation QuantileError @ 0.2 (on original scale): {final_error:,.2f}")

# Also, let's look at the predictions themselves to see if they are non-zero
print("\nSample of predictions:")
print(val_predictions[:20])
# Create output directories if they don't exist
outputs_dir = Path('outputs')
graph_dir = outputs_dir / 'graph'
model_dir = outputs_dir / 'model'
graph_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# Save the final model
model_path = model_dir / 'best_lgbm_model.joblib'
joblib.dump(final_model, model_path)
print(f"Final model saved to: {model_path}")

# Plot and save feature importance
plt.figure(figsize=(10, 8))
lgb.plot_importance(final_model, max_num_features=len(features_to_use), height=0.8)
plt.title("Feature Importance of Final Model")
plt.tight_layout()
importance_path = graph_dir / 'final_feature_importance.png'
plt.savefig(importance_path)
print(f"Feature importance plot saved to: {importance_path}")
