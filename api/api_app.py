from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import ModelLoadError, load_model, predict_window
from schemas import PredictRequest, PredictResponse

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models_out_final_v14"))

app = FastAPI(title="IIoT Mini Deploy", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=64)
def _cached_load(modality: str, load_nm: int, model_key: Optional[str], artifact_file: Optional[str], models_dir_str: str):
    return load_model(models_dir=models_dir_str, modality=modality, load_nm=load_nm, model_key=model_key, artifact_file=artifact_file)

@app.get("/health")
def health():
    return {"status": "ok", "models_dir": str(MODELS_DIR)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        lm = _cached_load(req.modality, int(req.load_nm), req.model_key, req.artifact_file, str(MODELS_DIR))
        out = predict_window(lm, req.samples, agg=req.agg)
        return out
    except ModelLoadError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
