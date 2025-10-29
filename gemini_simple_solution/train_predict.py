#!/usr/bin/env python

"""
This script trains a model and generates predictions for the TDT4173 task.

It works by:
1. Loading the pre-generated training data and test features.
2. Training a LightGBM regression model using 0.2 Quantile Loss.
3. Using this model to predict the values for the test set.
4. Saving the result to `submission.csv`.
"""

import pandas as pd
import numpy as np
import pathlib
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# --- Get the directory where this script is located ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()

print("Script started...")

# --- 1. Load Data ---

def load_prepared_data():
    """
    Loads the pre-generated training and test sets.
    """
    print("Loading prepared data...")
    try:
        training_data_file = BASE_DIR / 'training_dataset.csv'
        test_features_file = BASE_DIR / 'test_features.csv'
        
        # The index was saved in the CSV, so we use index_col=0
        training_data = pd.read_csv(training_data_file, index_col=0)
        X_test = pd.read_csv(test_features_file, index_col=0)
        X_test.index.name = 'ID'
        
        X_train = training_data.drop('target', axis=1)
        y_train = training_data['target']
        
        return X_train, y_train, X_test

    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
        print("Please run the `data_preparation.py` script first to generate the necessary files.")
        exit(1)

# --- 2. Main Execution ---

def main():
    # 1. Load data
    X_train, y_train, X_test = load_prepared_data()

    # 2. Train Model
    print("Training model...")
    
    # Create validation set
    X_train_full, X_val, y_train_full, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # Convert rm_id to 'category' dtype for LightGBM
    X_train_full['rm_id'] = X_train_full['rm_id'].astype('category')
    X_val['rm_id'] = pd.Categorical(
        X_val['rm_id'],
        categories=X_train_full['rm_id'].cat.categories
    )
    
    # Align test categories with train categories
    X_test['rm_id'] = pd.Categorical(
        X_test['rm_id'],
        categories=X_train_full['rm_id'].cat.categories
    )

    # Set objective to quantile
    model = LGBMRegressor(
        objective='quantile',
        alpha=0.2,
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        n_jobs=-1,
        categorical_feature=['rm_id']
    )
    
    print("Starting model fitting...")
    # Use early stopping
    model.fit(
        X_train_full, y_train_full,
        eval_set=[(X_val, y_val)],
        eval_metric='quantile',
        callbacks=[lgb.early_stopping(100, verbose=True)]
    )
    print("Model training complete.")

    # 3. Generate Predictions
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # We can't have negative weight
    predictions[predictions < 0] = 0
    
    # 4. Save Submission
    submission = pd.DataFrame({
        'ID': X_test.index,
        'predicted_weight': predictions
    })
    
    submission_filename = BASE_DIR / 'submission.csv'
    submission.to_csv(submission_filename, index=False)
    
    print("-" * 30)
    print(f"Success! Predictions saved to {submission_filename}")
    print(f"Submission file has {len(submission)} rows.")
    print(submission.head())
    print("-" * 30)

if __name__ == "__main__":
    main()
