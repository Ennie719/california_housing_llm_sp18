from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path):
    """Load config yaml file"""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)

def load_data(target=None):
    """Return features, or the requested target column, from the dataset."""
    config = load_config("configs/config.yaml")

    df = fetch_california_housing(
        as_frame=config["data"]["as_frame"]
    ).frame

    if target is None:
        return df.drop(columns="MedHouseVal")

    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in the dataset.")

    return df[target]

    # df = load_data()
def check_missing_values(df):
    """
    Check for missing values in the DataFrame.
    Returns:
    A boolean indicating if there are missing values.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    return df.isnull().values.any()
def check_duplicates(df):
    """
    Check for duplicate rows in the DataFrame.
    Returns:
    A boolean indicating if there are duplicate rows.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    return df.duplicated().any()

def splitting_data(df, test_size=None, random_state=None):
    """
    Split the input DataFrame into training and validation data.
    Returns:
    Training and validation features and targets.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    config = load_config("configs/config.yaml")
    if test_size is None:
        test_size = config["test_size"]
    if random_state is None:
        random_state = config["random_state"]
    target_column = "MedHouseVal"
    if target_column in df.columns:
        features = df.drop(columns=target_column)
        target = df[target_column]
    else:
        features = df
        target = load_data(target_column)
        if not features.index.equals(target.index):
            raise ValueError(
                "Feature rows must align with the California housing target."
            )

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test
def scale_features(x_train, x_test):
    """
    Scale the features using StandardScaler.
    Returns:
    Scaled training and validation features.
    """
    if x_train is None or x_test is None:
        raise ValueError("Training or test features are None. Please provide valid inputs.")
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled
def feature_target_corr(df):
    """
    Calculate the correlation between features and the target variable.
    Returns:
    A Series containing features with positive correlation to the target.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")

    target_column = "MedHouseVal"
    if target_column in df.columns:
        target = df[target_column]
        features = df.drop(columns=target_column)
    else:
        target = load_data(target_column)
        features = df
        if not features.index.equals(target.index):
            raise ValueError(
                "Feature rows must align with the California housing target."
            )

    correlations = features.corrwith(target)
    return correlations[correlations > 0.1].sort_values(ascending=False)
if __name__ == "__main__":
    split = splitting_data(load_data())
    print(split)