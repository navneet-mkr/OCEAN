import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List

from ocean.models.ocean import OCEAN
from ocean.utils.data_loader import OCEANDataLoader
from ocean.config.env import env
from ocean.utils.logging import get_logger
from .schemas import InferenceRequest, InferenceResponse, RootCause

logger = get_logger(__name__)

app = FastAPI(
    title="OCEAN RCA API",
    description="API for Online Multi-modal Root Cause Analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: OCEAN = None
data_loader: OCEANDataLoader = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, data_loader
    try:
        # Initialize data loader
        data_loader = OCEANDataLoader(
            data_dir=str(env.DATA_DIR),
            device=env.DEVICE
        )
        
        # Initialize model
        model = OCEAN(
            n_entities=data_loader.n_entities,
            n_metrics=data_loader.n_metrics,
            n_logs=data_loader.n_logs
        ).to(env.DEVICE)
        
        # Load model checkpoint
        checkpoint_path = env.OUTPUT_DIR / "checkpoints" / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Perform online root cause analysis
    """
    try:
        # Convert request data to DataFrames
        metrics_df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'entity_id': m.entity_id,
                **m.values
            } for m in request.metrics
        ])
        
        logs_df = pd.DataFrame([
            {
                'timestamp': log.timestamp,
                'entity_id': log.entity_id,
                'message': log.message,
                'level': log.level
            } for log in request.logs
        ])
        
        # Process data
        metrics_tensor, logs_tensor, _ = data_loader.process_streaming_batch(
            metrics_df, logs_df, 1.0  # Assuming anomaly is represented by 1.0
        )
        
        # Get previous adjacency matrix or create new one
        if request.previous_adj_matrix:
            old_adj = torch.tensor(request.previous_adj_matrix).to(env.DEVICE)
        else:
            old_adj = torch.zeros(
                (model.n_entities, model.n_entities),
                device=env.DEVICE
            )
        
        # Get entity index from ID
        try:
            kpi_idx = metrics_df['entity_id'].unique().tolist().index(request.kpi_entity_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"KPI entity ID {request.kpi_entity_id} not found in data"
            )
        
        # Perform inference
        with torch.no_grad():
            ranked_indices, ranked_probs, should_stop, _ = model(
                metrics=metrics_tensor,
                logs=logs_tensor,
                old_adj=old_adj,
                kpi_idx=kpi_idx,
                is_training=False
            )
        
        # Convert results to response format
        entity_ids = metrics_df['entity_id'].unique().tolist()
        root_causes = [
            RootCause(
                entity_id=entity_ids[idx],
                probability=float(prob)
            )
            for idx, prob in zip(ranked_indices, ranked_probs)
        ]
        
        # Calculate confidence as average of top probabilities
        confidence = float(np.mean(ranked_probs))
        
        return InferenceResponse(
            root_causes=root_causes,
            should_stop=should_stop,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None} 