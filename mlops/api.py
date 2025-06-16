import os
import logging
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

# from mlops.model_registry import ModelRegistry
from mlops.schema import CustomerFeatures, PredictionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
MODEL_REQUEST_COUNT = Counter('model_requests_total', 'Total model requests', ['model_name', 'version', 'status'])
MODEL_REQUEST_LATENCY = Histogram('model_request_duration_seconds', 'Model request latency')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total predictions made', ['model_name', 'prediction'])
DRIFT_SCORE = Gauge('model_drift_score', 'Current drift score', ['model_name', 'drift_type'])

# FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction",
    version="0.1.0"
)

# Global model registry
# model_registry = ModelRegistry()
current_model = None
current_metadata = None

def load_model():
    global current_model, current_metadata
    try:
        model_name = os.getenv("MODEL_NAME", "churn_model")
        
        # current_model, current_metadata = model_registry.get_model(name=model_name)
        logger.info(f"Loaded model: {current_metadata.name} v{current_metadata.version}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def get_model():
    if current_model is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model not available")
    return current_model, current_metadata

# @app.lifespan("startup")
# async def startup_event():
#     logger.info("Starting Churn Prediction API")
#     if not load_model():
#         logger.warning("Failed to load model on startup")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": current_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures, background_tasks: BackgroundTasks, model_data = Depends(get_model)):
    """Predict churn for a single customer"""
    try:
        # Prepare features
        # Make prediction
        # Calculate confidence
        # Create response
        # Record metrics
        # Log prediction (background task)
        return {} # response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def start_server():
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("mlops.api:app", host="0.0.0.0", port=port, reload=True)