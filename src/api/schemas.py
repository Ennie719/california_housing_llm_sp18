from pydantic import BaseModel, Field
from typing import Optional

class HousePredictionRequest(BaseModel):
    """Schema for house prediction request - validates all 8 features"""

    # Property listing features
    MedInc: float = Field(..., description="Median income in the area")
    AveOccup: float = Field(..., description="Average occupancy in the area")
    Longitude: float = Field(..., description="Longitude of the property")
    Latitude: float = Field(..., description="Latitude of the property")
    HouseAge: float = Field(..., description="Age of the house")
    AveRooms: float = Field(..., description="Average number of rooms in the area")
    Population: float = Field(..., description="Population in the area")
    AveBedrms: float = Field(..., description="Average number of bedrooms in the area")

    class Config:
        schema_extra = {
            "example": {
                "MedInc": 8.5,
                "AveOccup": 3.2,
                "Longitude": -74.0060,
                "Latitude": 40.7128,
                "HouseAge": 20,
                "AveRooms": 5.0,
                "Population": 1500,
                "AveBedrms": 2.5
            }
        }

# YOUR TASK: Add the PredictionResponse schema
# This schema defines what your API returns after making a prediction
# It should have:
# - predicted_price: float (the predicted house price)
# - currency: str with default value "USD"
# - model_version: str (the version of the model used)

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    predicted_price: float = Field(..., description="Predicted house price")
    currency: str = Field("USD", description="Currency of the predicted price")
    model_version: Optional[str] = Field(None, description="Version of the model used for prediction")
class ModelInfoResponse(BaseModel):
    # Add the three fields here
    # Remember to use Field() to add descriptions!
     # Remove this line when you add the fields
# YOUR TASK: Add the ModelInfoResponse schema
# This schema defines what the /model/info endpoint returns
# It should have:
# - model_type: str
    model_type: str = Field(..., description="Type of the model (e.g., RandomForest)")
# - version: str
    version: str = Field(..., description="Version of the model")
# - features: list of strings
    features: list[str] = Field(..., description="List of features used by the model")  
# - training_date: str
    training_date: str = Field(..., description="Date when the model was trained")
# - rmse: float 
    rmse: float = Field(..., description="Root Mean Squared Error of the model on test data")
# - description: str
    description: str = Field(..., description="Description of the model and its performance")
    
    # Add all six fields here
    # Remember to use Field() to add descriptions!
    # pass  # Remove this line when you add the fields

# YOUR TASK: Add the HealthCheckResponse schema
# This schema defines what the /health endpoint returns
class HealthCheckResponse(BaseModel):
    # It should have:
    """Schema for health check response"""
    # - status: str (should be "healthy" or "unhealthy")
    status: str = Field(..., description="Health status of the service (e.g., 'healthy' or 'unhealthy')")
# - model_loaded: bool (True if model is loaded, False otherwise)
    model_loaded: bool = Field(..., description="Indicates if the model is loaded successfully")
# - message: str (a descriptive message)
    message: str = Field(..., description="Descriptive message about the health status")
    # Add the three fields here