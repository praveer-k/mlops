
from typing import Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator

class CustomerFeatures(BaseModel):
    """Customer features for prediction"""
    customer_id: str
    monthly_charges: float = Field(ge=0)
    tenure: int = Field(ge=0)
    total_charges: float = Field(ge=0)
    high_value_fiber: bool
    churn: bool
    
    @field_validator("customer_id")
    def normalize_customer_id(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="before")
    def validate_total_charges(cls, values: dict[str, Any]) -> dict:
        tenure = values.get("tenure")
        monthly = values.get("monthly_charges")
        total = values.get("total_charges")
        if tenure is not None and monthly is not None and total is not None:
            expected_min_total = tenure * monthly
            if total < expected_min_total * 0.9:
                raise ValueError(
                    f"Total charges ({total}) seem too low for tenure * monthly_charges ({expected_min_total})"
                )
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "monthly_charges": 65.50,
                "tenure": 24,
                "total_charges": 1572.00,
                "high_value_fiber": True,
                "churn": False
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    confidence: str
    model_version: str
    prediction_timestamp: datetime

class ModelStatus(BaseModel):
    """Model status information"""
    model_name: str
    version: str
    stage: str
    loaded_at: datetime
    total_predictions: int
    avg_latency_ms: float
    drift_scores: Dict[str, float]