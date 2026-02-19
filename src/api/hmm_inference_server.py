#!/usr/bin/env python3
"""
HMM Inference Server
====================

Standalone FastAPI app for HMM regime inference running on Contabo VPS.
Provides endpoints for regime detection, model version info, and model push.

Runs on port 8001.

Usage:
    python src/api/hmm_inference_server.py
    # or
    uvicorn src.api.hmm_inference_server:app --host 0.0.0.0 --port 8001
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
from fastapi import FastAPI, APIRouter, Depends, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hmm_inference_server")

# Load configuration
CONFIG_PATH = os.environ.get("HMM_CONFIG_PATH", "config/hmm_config.json")


def load_config() -> Dict:
    """Load HMM configuration."""
    config_file = Path(project_root) / CONFIG_PATH
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


config = load_config()

# Model paths
CONTABO_MODEL_PATH = os.environ.get("CONTABO_MODEL_PATH", "/data/hmm/models")
CONTABO_METADATA_PATH = os.environ.get("CONTABO_METADATA_PATH", "/data/hmm/metadata")

# API Key
CONTABO_HMM_API_KEY = os.environ.get("CONTABO_HMM_API_KEY", "")


# ============= API Key Dependency =============


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key for protected endpoints."""
    if not CONTABO_HMM_API_KEY:
        logger.warning("CONTABO_HMM_API_KEY not set - API key verification disabled")
        return x_api_key
    
    if x_api_key != CONTABO_HMM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key


# ============= Pydantic Models =============


class RegimeResponse(BaseModel):
    """Regime detection response."""
    symbol: str
    regime: str
    confidence: float
    model_version: str


class ModelVersionResponse(BaseModel):
    """Model version information."""
    version: str
    checksum: str
    training_date: str
    model_type: str
    symbol: Optional[str] = None


class PushResponse(BaseModel):
    """Model push response."""
    status: str
    message: str


# ============= Router =============


router = APIRouter(prefix="/api/hmm", tags=["hmm"])


def _load_model(model_type: str = "universal", symbol: Optional[str] = None):
    """
    Load HMM model from disk.
    
    Args:
        model_type: Type of model (universal, per_symbol)
        symbol: Symbol for per-symbol models
        
    Returns:
        Tuple of (model, metadata, regime_mapping)
    """
    import pickle
    from src.risk.physics.hmm_features import load_config_from_file
    
    # Build model filename
    if model_type == "universal":
        model_filename = "hmm_universal_*.pkl"
    elif model_type == "per_symbol" and symbol:
        model_filename = f"hmm_per_symbol_{symbol}_*.pkl"
    else:
        model_filename = "hmm_universal_*.pkl"
    
    # Find latest model file
    model_dir = Path(CONTABO_MODEL_PATH)
    if not model_dir.exists():
        return None, None, None
    
    model_files = list(model_dir.glob(model_filename))
    if not model_files:
        # Try universal as fallback
        model_files = list(model_dir.glob("hmm_universal_*.pkl"))
    
    if not model_files:
        return None, None, None
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Load model
    with open(latest_model, 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    metadata_file = latest_model.with_suffix(latest_model.suffix + ".metadata.json")
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    
    # Load regime mapping from config
    regime_mapping = config.get("regime_mapping", {
        "0": "TRENDING_LOW_VOL",
        "1": "TRENDING_HIGH_VOL",
        "2": "RANGING_LOW_VOL",
        "3": "RANGING_HIGH_VOL"
    })
    
    return model, metadata, regime_mapping


def _get_features_for_prediction(symbol: str) -> Optional[Any]:
    """
    Get the most recent features for regime prediction.
    
    In a production system, this would fetch from the WARM tier database.
    For now, we'll generate synthetic features for demonstration.
    """
    import numpy as np
    
    # Try to get real features from database
    try:
        from src.database.duckdb_connection import DuckDBConnection
        
        warm_db_path = os.environ.get("WARM_DB_PATH", "/data/market_data.duckdb")
        
        with DuckDBConnection(db_path=warm_db_path) as conn:
            result = conn.execute_query(f"""
                SELECT open, high, low, close, volume
                FROM market_data
                WHERE symbol = '{symbol}' AND timeframe = 'H1'
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            
            if result and result.fetchone():
                df = result.df()
                if len(df) >= 50:
                    # Calculate features from actual data
                    close_prices = df['close'].values
                    returns = np.diff(np.log(close_prices))
                    volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
                    
                    # Create feature vector
                    features = np.array([
                        np.mean(returns[-10:]) if len(returns) >= 10 else 0,
                        volatility,
                        np.mean(volatility) if len(returns) >= 20 else volatility,
                        close_prices[-1] / close_prices[-10] - 1 if len(close_prices) >= 10 else 0
                    ])
                    return features.reshape(1, -1)
    except Exception as e:
        logger.warning(f"Failed to fetch features from database: {e}")
    
    # Fallback: generate synthetic features for demonstration
    import numpy as np
    np.random.seed(42)
    features = np.random.randn(1, 4) * 0.1
    return features


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "vps": "contabo",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/regime", response_model=Dict[str, Dict])
async def get_all_regimes():
    """
    Get regime for all configured primary symbols.
    
    Returns regime, confidence, and model version for each symbol.
    """
    symbols = config.get("symbols", {}).get("primary", ["EURUSD", "GBPUSD", "XAUUSD"])
    
    results = {}
    for symbol in symbols:
        try:
            # Get features for prediction
            features = _get_features_for_prediction(symbol)
            
            if features is None:
                results[symbol] = {
                    "regime": "UNKNOWN",
                    "confidence": 0.0,
                    "model_version": "N/A",
                    "error": "No features available"
                }
                continue
            
            # Load model
            model, metadata, regime_mapping = _load_model("per_symbol", symbol)
            
            if model is None:
                # Fallback to universal model
                model, metadata, regime_mapping = _load_model("universal")
            
            if model is None:
                results[symbol] = {
                    "regime": "NO_MODEL",
                    "confidence": 0.0,
                    "model_version": "N/A",
                    "error": "No trained model available"
                }
                continue
            
            # Predict
            try:
                hidden_states = model.predict(features)
                regime_idx = hidden_states[0]
                regime = regime_mapping.get(str(regime_idx), f"UNKNOWN_{regime_idx}")
                
                # Calculate confidence (simplified - use model score)
                score = model.score(features)
                confidence = min(1.0, max(0.0, 1.0 - abs(score) / 1000))
                
                results[symbol] = {
                    "regime": regime,
                    "confidence": float(confidence),
                    "model_version": metadata.get("version", "unknown"),
                    "model_type": metadata.get("model_type", "unknown")
                }
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                results[symbol] = {
                    "regime": "ERROR",
                    "confidence": 0.0,
                    "model_version": metadata.get("version", "unknown"),
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"Failed to get regime for {symbol}: {e}")
            results[symbol] = {
                "regime": "ERROR",
                "confidence": 0.0,
                "model_version": "N/A",
                "error": str(e)
            }
    
    return results


@router.get("/regime/{symbol}", response_model=RegimeResponse)
async def get_regime_for_symbol(symbol: str):
    """
    Get regime for a specific symbol.
    
    Args:
        symbol: Trading symbol (e.g., EURUSD, GBPUSD)
        
    Returns:
        Regime information with confidence and model version
    """
    symbol = symbol.upper()
    
    # Get features for prediction
    features = _get_features_for_prediction(symbol)
    
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features available for symbol {symbol}"
        )
    
    # Load model
    model, metadata, regime_mapping = _load_model("per_symbol", symbol)
    
    if model is None:
        # Fallback to universal model
        model, metadata, regime_mapping = _load_model("universal")
    
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model available for symbol {symbol}"
        )
    
    # Predict
    try:
        hidden_states = model.predict(features)
        regime_idx = hidden_states[0]
        regime = regime_mapping.get(str(regime_idx), f"UNKNOWN_{regime_idx}")
        
        # Calculate confidence
        score = model.score(features)
        confidence = min(1.0, max(0.0, 1.0 - abs(score) / 1000))
        
        return RegimeResponse(
            symbol=symbol,
            regime=regime,
            confidence=float(confidence),
            model_version=metadata.get("version", "unknown")
        )
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/model/version", response_model=ModelVersionResponse)
async def get_model_version():
    """
    Get the latest model version information.
    
    Returns version, checksum, training date, and model type.
    """
    model_dir = Path(CONTABO_MODEL_PATH)
    metadata_dir = Path(CONTABO_METADATA_PATH)
    
    # Find latest metadata file
    metadata_files = list(metadata_dir.glob("*.json")) if metadata_dir.exists() else []
    
    if not metadata_files:
        # Try model directory
        metadata_files = list(model_dir.glob("*.metadata.json"))
    
    if not metadata_files:
        raise HTTPException(
            status_code=404,
            detail="No model metadata found"
        )
    
    latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_metadata) as f:
        metadata = json.load(f)
    
    return ModelVersionResponse(
        version=metadata.get("version", "unknown"),
        checksum=metadata.get("checksum", "unknown"),
        training_date=metadata.get("training_date", "unknown"),
        model_type=metadata.get("model_type", "unknown"),
        symbol=metadata.get("symbol")
    )


@router.post("/model/push", response_model=PushResponse)
async def push_model(background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    """
    Push model from Contabo to Cloudzy.
    
    Requires X-API-Key header with valid API key.
    Sync runs in background.
    """
    def _sync_task():
        """Background task to sync model."""
        try:
            from src.router.hmm_version_control import HMMVersionControl
            
            logger.info("Starting model sync in background...")
            vc = HMMVersionControl()
            
            # Sync universal model
            universal_success = vc.sync_model(model_type="universal")
            
            # Sync per-symbol models
            symbols = config.get("symbols", {}).get("primary", [])
            for symbol in symbols:
                vc.sync_model(model_type="per_symbol", symbol=symbol)
            
            logger.info(f"Model sync completed: universal={universal_success}")
            
        except Exception as e:
            logger.error(f"Model sync failed: {e}")
    
    # Schedule background task
    background_tasks.add_task(_sync_task)
    
    return PushResponse(
        status="accepted",
        message="Model sync initiated in background"
    )


# ============= Create App =============


app = FastAPI(
    title="HMM Inference Server",
    description="HMM Regime Detection API running on Contabo VPS",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "HMM Inference Server",
        "vps": "contabo",
        "port": 8001,
        "status": "running"
    }


# ============= Entry Point =============


if __name__ == "__main__":
    logger.info("Starting HMM Inference Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
