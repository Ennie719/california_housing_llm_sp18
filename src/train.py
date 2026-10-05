from logging import config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yaml
import mlflow
import mlflow.sklearn
from src.preprocess import load_config, load_data, splitting_data
import json
import os
import joblib
    
def model_training(df, config):
    """
    Train a Random Forest Regressor on the training data.
    Returns:
    The trained model.
    # """
    x_train,x_test, y_train,y_test = splitting_data(df, 
                                                    test_size=config["test_size"], 
                                                    random_state=config["random_state"]
                                                    )

    rf = RandomForestRegressor(n_estimators = config["model"]["n_estimators"],
                                max_depth = config["model"]["max_depth"],
                                min_samples_split = config["model"]["min_samples_split"]
                                )
    rf_fit = rf.fit(x_train, y_train)
    return rf_fit

def feature_importance(rf, feature_names):
    """
    Calculate feature importance from the fitted Random Forest model.
    Returns:
    DataFrame containing features and their importance scores.
    """
    if rf is None or feature_names is None:
        raise ValueError("Model or feature names are None. Please provide valid inputs.")

    importances_val = rf.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances_val,
        'importance_Percent': importances_val * 100
    }).sort_values('importance', ascending=False)

    return importance_df

def trial_experiment():
    config = load_config("configs/config.yaml")
    df = load_data()
    x_train, x_test, y_train, y_test = splitting_data(df)
    best_mse = float("inf")
    model_dir = Path("models/expirement_models_joblib")
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("California Housing")
    for exp in config["experiments"]:
        print(f"Starting {exp['name']}")
        print("=" * 60)
        with mlflow.start_run(run_name=exp["name"]):
            model = model_training(
                df,
                exp
            )
            predictions = model.predict(x_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"mse": mse, "r2": r2})
            mlflow.sklearn.log_model(model, "model")
            print(f"{exp['name']}: MSE={mse:.4f}, R2={r2:.4f}, predictions ={predictions} ")
            joblib.dump(model, f"models/expirement_models_joblib/model_{exp['name']}.joblib" )
            if mse < best_mse:
                best_mse = mse
                joblib.dump(model, model_dir / "best_model.joblib")
            joblib.dump(predictions, f"models/expirement_models_joblib/predictions_{exp['name']}.joblib",  )
            joblib.dump(mse, f"models/expirement_models_joblib/mse_{exp['name']}.joblib")
            joblib.dump(r2, f"models/expirement_models_joblib/r2_{exp['name']}.joblib")
    return None

def main():
    from src.evaluate import model_evaluation

    config = load_config("configs/config.yaml")
    df = load_data()
    x_train, x_valid, y_train, y_valid = splitting_data(
        df,
        config["test_size"],
        config["random_state"],
    )
    model = model_training(df, config)

    mlflow.set_experiment("California Housing")
    with mlflow.start_run():
        mse, r2 = model_evaluation(model, x_valid, y_valid)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"mse": mse, "r2": r2})
        mlflow.sklearn.log_model(model, "model")
    # return df
    stats = {"mse": mse, "r2": r2}

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/results.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    #  with open("configs/config.yaml") as f:
        # config = yaml.safe_load(f)
        # df = load_data()
        # rf = model_training(df, config)
    trial_experiment()