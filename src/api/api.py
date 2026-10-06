import joblib
import numpy as np
import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# from api.schemas import PredictionResponse
# Global variable to store the loaded model
# model = joblib.load("../models/expirement_models_joblib/model_Trial_1.joblib")
model = None
metadata = None

def load_model_and_metadata():
    global model, metadata

    try:
        project_root = Path(__file__).resolve().parents[2]
        model = joblib.load(
            project_root / "models/expirement_models_joblib/best_model.joblib"
        )
        if not callable(getattr(model, "predict", None)):
            raise ValueError("Saved model artifact does not contain a predictor")
        if getattr(model, "n_outputs_", None) != 1:
            raise ValueError(
                "Saved model must be a single-output regressor; retrain and replace it."
            )

        with (project_root / "models/model_metadata.json").open("r") as f:
            metadata = json.load(f) # Replace this with json.load(...)
        if set(metadata["features"]) != set(model.feature_names_in_):
            raise ValueError(
                "Model metadata features do not match the saved model features."
            )

        print("Model and metadata loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        return False
load_model_and_metadata()

def make_prediction(data: Dict[str, Any]) -> float:
    global model
   
    if model is None or metadata is None:
        raise ValueError("Model not loaded")
    feature_values = {
        feature_name: data[feature_name] for feature_name in model.feature_names_in_
    }
    X = pd.DataFrame([feature_values])
    print(X)

    # prediction = model.predict(X)[0]  # Replace with actual prediction
    raw = np.asarray(model.predict(X))
    if raw.size != 1:
        raise ValueError(
            f"Expected one prediction, got shape {raw.shape}. "
            "The model may have been trained with multiple targets."
        )
    prediction = float(raw.ravel()[0])
    return round(prediction, 2)
# data = load_model_and_metadata()
# predict = make_prediction(data)
# print(predict)
def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        Dictionary containing model metadata

    This function should simply return the loaded metadata dictionary.
    If metadata is not loaded, it should raise an error.
    """
    global metadata

    # YOUR TASK: Check if metadata is loaded and return it
    # If metadata is None, raise ValueError("Model metadata not loaded")
    if metadata is None:
        raise ValueError("Model metadata not loaded")
    return metadata
# Replace with your implementation

def check_health() -> Dict[str, Any]:
    """
    Check the health status of the service.

    Returns:
        Dictionary with health status information

    This function should:
    1. Check if model is loaded (model is not None)
    2. Check if metadata is loaded (metadata is not None)
    3. Return a dictionary with status information
    """
    global model, metadata
    # YOUR TASK: Create health status dictionary
    # Include:
    # - "status": "healthy" if both model and metadata are loaded, "unhealthy" otherwise
    # - "model_loaded": True/False based on whether model is not None
    # - "message": Descriptive message about the status
    model_loaded = model is not None
    metadata_loaded = metadata is not None
 
    health_status = {
        "status": "healthy" if model_loaded and metadata_loaded else "unhealthy",
        "model_loaded": model_loaded,
        # "metadata_loaded": metadata_loaded,
        "message": "Service is healthy and model is loaded" if model_loaded and metadata_loaded else "Service is unhealthy or model is not loaded"
    }
    return health_status
    
   