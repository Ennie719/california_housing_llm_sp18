import sys
import numpy as np
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import load_data, splitting_data, check_missing_values, check_duplicates, scale_features, feature_target_corr
from sklearn.metrics import mean_squared_error, r2_score
from src.train import model_training, feature_importance
from sklearn.datasets import fetch_california_housing

def test_data_load():
    df = load_data()
    assert not df.empty, "DataFrame is empty. Data loading failed."
    
# def test_encode_categoricals():
#     """Test the one-hot encoding of categorical columns."""
#     df = load_data()
#     encoded_df = encode_categoricals(df, ['HouseAge'])
#     assert not encoded_df.empty, "One-hot encoding failed."
#     assert 'HouseAge_10.0' in encoded_df.columns, "One-hot encoding did not create expected columns."
def test_missing_values_check():
    """Test the check for missing values in the DataFrame."""
    df = load_data()
    
    has_missing = check_missing_values(df)
    assert not has_missing, "DataFrame contains missing values."
def test_duplicates_check():
    """Test the check for duplicate rows in the DataFrame."""
    df = load_data()
    has_duplicates = check_duplicates(df)
    assert not has_duplicates, "DataFrame contains duplicate rows."
def test_splitting_data():
    """Test the splitting of the DataFrame into training and validation sets."""
    df = load_data()
    x_train, x_test, y_train, y_test = splitting_data(df)
    assert len(x_train) > 0 and len(x_test) > 0, "Data splitting failed."
    assert len(y_train) > 0 and len(y_test) > 0, "Data splitting failed."
# df = load_data()
# df_split = splitting_data(df)
# print(f"Training set size: {len(df_split[0])}, Validation set size: {len(df_split[1])}")
def test_feature_target_corr():
    """Test the calculation of feature-target correlation."""
    df = load_data()
    corr_df = feature_target_corr(df)
    assert not corr_df.empty, "Feature-target correlation calculation failed."
    # assert 'MedHouseVal' in corr_df, "Correlation DataFrame does not contain target column."
df = load_data()
corr_df = feature_target_corr(df)
print(corr_df)