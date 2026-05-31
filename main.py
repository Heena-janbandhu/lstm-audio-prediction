"""
LSTM Audio Sequence Predictor — FastAPI (Render Deployment)
============================================================
POST /predict  →  upload audio file, get predicted MFCC frames
"""

import io, json, pickle, tempfile, os
from pathlib import Path
from typing import List

import numpy as np
import librosa
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# ── Model paths ────────────────────────────────────────────────────────
MODEL_DIR   = Path("saved_model")
CONFIG_PATH = MODEL_DIR / "config.json"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH  = MODEL_DIR / "lstm_audio_model.keras"

_model = _scaler = _config = None

def get_config():
    global _config
    if _config is None:
        with open(CONFIG_PATH) as f:
            _config = json.load(f)
    return _config

def get_scaler():
    global _scaler
    if _scaler is None:
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
    return _scaler

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        print("✅ LSTM model loaded")
    return _model

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LSTM Audio Sequence Predictor",
    description="Predicts the next MFCC frames from a short audio clip using a trained LSTM model.",
    version="1.0.0",
)

# ── Schemas ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_frames: List[List[float]]
    pred_len: int
    n_mfcc: int
    input_duration_sec: float
    message: str

class HealthResponse(BaseModel):
    status: str
    model_ready: bool

# ── Helpers ────────────────────────────────────────────────────────────
def extract_mfcc(audio_bytes: bytes, filename: str):
    cfg = get_config()
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=cfg["target_sr"], mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    mfccs = librosa.feature.mfcc(
        y=y, sr=cfg["target_sr"], n_mfcc=cfg["n_mfcc"],
        hop_length=cfg["hop_length"], n_fft=cfg["n_fft"]
    ).T
    return mfccs, len(y) / cfg["target_sr"]

# ── Routes ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "LSTM Audio Predictor is live!", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_ready=MODEL_PATH.exists())

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="WAV / MP3 / OGG audio file")):
    """Upload an audio clip → receive predicted next MFCC frames."""
    cfg    = get_config()
    scaler = get_scaler()
    model  = get_model()

    seq_len, pred_len, n_mfcc = cfg["seq_len"], cfg["pred_len"], cfg["n_mfcc"]

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    mfccs, duration = extract_mfcc(audio_bytes, file.filename or "audio.wav")

    if len(mfccs) < seq_len:
        raise HTTPException(
            status_code=422,
            detail=f"Audio too short ({len(mfccs)} frames). Need at least {seq_len} frames (~1 second)."
        )

    input_norm = scaler.transform(mfccs[-seq_len:])
    X          = input_norm[np.newaxis, ...]
    pred_flat  = model.predict(X, verbose=0)
    pred_orig  = scaler.inverse_transform(pred_flat.reshape(pred_len, n_mfcc))

    return PredictionResponse(
        predicted_frames=pred_orig.tolist(),
        pred_len=pred_len,
        n_mfcc=n_mfcc,
        input_duration_sec=round(duration, 3),
        message=f"Predicted {pred_len} future MFCC frames from {duration:.2f}s of audio."
    )

# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
