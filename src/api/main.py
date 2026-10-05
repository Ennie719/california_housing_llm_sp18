print("=== SERVER STARTING ===")
import json
from fastapi import FastAPI, HTTPException
from .schemas import (
    HousePredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthCheckResponse
)
from .api import (
    load_model_and_metadata,
    make_prediction,
    get_model_info,
    check_health
)

# Create FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="Machine learning service for predicting US apartment prices based on 13 features",
    version="1.0.0"
)

# YOUR TASK: Load model at startup
# Use the @app.on_event("startup") decorator to load the model when the service starts
# This ensures the model is loaded ONCE, not on every request
@app.on_event("startup")
async def startup_event():
    """Load model and metadata when the service starts"""
    # startup = load_model_and_metadata()
    # Call load_model_and_metadata() from api.py
    # If it returns False, print an error message
    success = load_model_and_metadata()  # Replace with actual function call
    if not success:
        print("WARNING: Failed to load model at startup")

# YOUR TASK: Implement the health check endpoint
# This endpoint should:
# 1. Call check_health() from api.py
# 2. Return a Hea
# lthCheckResponse with the health information
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check if the service is healthy and model is loaded.

    Returns:
        HealthCheckResponse with current service status
    """
        # Get health status from api.py
    health_status  = check_health()
    # Create and return HealthCheckResponse
    return  HealthCheckResponse( status=health_status["status"], model_loaded=health_status["model_loaded"], message=health_status["message"] )  # Replace with your implementation
# YOUR TASK: Implement the model info endpoint
# This endpoint should:
# 1. Call get_model_info() from api.py
# 2. Handle the case where metadata is not loaded (raise HTTPException with status_code=503)
# 3. Return a ModelInfoResponse with the model information
@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        ModelInfoResponse with model metadata

    Raises:
        HTTPException: If model metadata is not loaded
    """
    try:
        model_info = get_model_info()
        # Get model info and return it
        return ModelInfoResponse( model_type=model_info["model_type"], version=model_info["version"], features=model_info["features"], training_date=model_info["training_date"], rmse=model_info["rmse"], description=model_info["description"] ) 
    except ValueError as e:
        # Raise HTTPException with status_code=503 (Service Unavailable)
        # Include the error message in the detail
        raise HTTPException(status_code=503, detail="Service Unavailable")  # Replace with proper error handling

# YOUR TASK: Implement the prediction endpoint
# This endpoint should:
# 1. Convert the Pydantic model to a dictionary
# 2. Call make_prediction() from api.py
# 3. Get model version from metadata
# 4. Return a PredictionResponse
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    print("Starting make_prediction ", request)
    print("Input data:", request)

    try:
        # YOUR TASK: Convert request to dictionary
        # Hint: Pydantic models have a .dict() method
        features_dict = request.dict() # Replace with request.dict()
        # print("Features dict:", features_dict)

        # YOUR TASK: Make prediction
        # int = int(make_prediction('predicted_price'))
        predicted_price = make_prediction(features_dict) # Replace with make_prediction() call
        # print(f"Type: {type(predicted_price)}")
        # print(f"Content: {predicted_price}")
        # YOUR TASK: Get model version from metadata
        # You'll need to call get_model_info() to get the version
        model_info = get_model_info()
        model_version = model_info["version"]
        # # Replace with actual version from metadata
        print("model_version extracted successfully:", model_version)
        # YOUR TASK: Create and return PredictionResponse
        
        return PredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            model_version=model_version
        )

    except ValueError as e:
        # Model not loaded error
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Root endpoint for basic information
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "House Price Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Check service health",
            "/model/info - Get model information",
            "/predict - Make price prediction",
            "/docs - Interactive API documentation"
        ]
    }