from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from mlp_core import DriftDetector
import json
from typing import List
import logging
from datetime import datetime
import pandas as pd
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLP Classification API")

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    prediction_classes: List[int]

class HealthResponse(BaseModel):
    status: str
    model_version: str
    last_updated: str

# Load model at startup
try:
    sess = ort.InferenceSession("production_model/model.onnx")
    input_name = sess.get_inputs()[0].name
    
    with open("production_model/metadata.json") as f:
        metadata = json.load(f)
    
    # Initialize drift detector with training data stats
    drift_detector = DriftDetector(np.random.randn(100, metadata['input_size']))  # In prod, load actual training data
    
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "last_updated": metadata['training_date']
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        input_data = np.array(request.features, dtype=np.float32)
        
        # Check for data drift
        drift_results = drift_detector.check_drift(input_data)
        if drift_results['overall_drift']:
            logger.warning(f"Data drift detected: {json.dumps(drift_results, indent=2)}")
        
        # Make prediction
        predictions = sess.run(None, {input_name: input_data})[0]
        
        return {
            "predictions": predictions.flatten().tolist(),
            "prediction_classes": (predictions > 0.5).astype(int).flatten().tolist()
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_csv")
async def predict_csv(file: bytes = File(...)):
    try:
        # Read CSV data
        df = pd.read_csv(io.BytesIO(file))
        input_data = df.values.astype(np.float32)
        
        # Check for data drift
        drift_results = drift_detector.check_drift(input_data)
        if drift_results['overall_drift']:
            logger.warning(f"Data drift detected in batch: {json.dumps(drift_results, indent=2)}")
        
        # Make predictions
        predictions = sess.run(None, {input_name: input_data})[0]
        
        # Prepare response
        df['prediction'] = predictions
        df['prediction_class'] = (predictions > 0.5).astype(int)
        
        return Response(
            content=df.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_metadata")
async def get_metadata():
    return metadata

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)