import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.preprocess import load_config, load_data, splitting_data
from src.train import model_training, feature_importance
from sklearn.datasets import fetch_california_housing
import pandas as pd
from src.evaluate import model_evaluation
n_rows = 200

df_sample = pd.DataFrame({
    "size_sqft": np.random.randint(500, 3500, n_rows),
    "num_rooms": np.random.randint(1, 6, n_rows),
    "age_years": np.random.randint(0, 50, n_rows),
})
df_sample["MedHouseVal"] = (
    df_sample["size_sqft"] * 150
    + df_sample["num_rooms"] * 5000
    - df_sample["age_years"] * 300
    + np.random.normal(0, 10000, n_rows)
).round(2)
config = load_config("configs/config.yaml")
sample_splitt_data = (df_sample)
sample_model = model_training(df_sample, config)
x_train, x_test, y_train, y_test = splitting_data(df_sample, test_size = config["test_size"], random_state=config["random_state"])
sample_model_eval = model_evaluation(sample_model, x_test, y_test)
# print(sample_model_eval)
def test_prediction_shape():
    predictions = sample_model.predict(x_test)
    assert np.issubdtype(predictions.dtype, np.number), f"Predictions have to be numbers"
print(test_prediction_shape())

