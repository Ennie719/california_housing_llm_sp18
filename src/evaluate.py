import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from src.preprocess import load_config, splitting_data, load_data
from src.train import model_training

def model_evaluation(model, x_test, y_test):
    """
    Evaluate the performance of a Random Forest Regressor on the test data.
    Returns:
    The trained model and its performance metrics.
    """
    predictions = model.predict(x_test)
    mse_rf_tuned = mean_squared_error(y_test, predictions)
    rmse_rf_tuned = np.sqrt(mse_rf_tuned)
    r2_rf_tuned = r2_score(y_test, predictions)
    # print("Metrics:"str)
    # print(f"MSE:  {mse_rf_tuned:.4f}")
    # # print(f"RMSE: {rmse_rf_tuned:.4f}")
    # print(f"R²:   {r2_rf_tuned:.1%}")
    # print(type(predictions))
    # print()
    return mse_rf_tuned, r2_rf_tuned
if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    df = load_data()
    model = model_training(df, config)
    x_train, x_test, y_train, y_test = splitting_data(df)
    eva = model_evaluation(model, x_test, y_test)
    print(eva)
    