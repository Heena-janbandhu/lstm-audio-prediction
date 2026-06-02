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
from fastapi.responses import HTMLResponse
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

# ── Homepage HTML ──────────────────────────────────────────────────────
HOMEPAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LSTM Audio Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #080c14;
    --surface: #0e1420;
    --border: #1e2d45;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --text: #e2eaf5;
    --muted: #5a7090;
    --success: #00e676;
    --error: #ff5252;
    --warn: #ffd740;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .glow-orb {
    position: fixed;
    width: 600px; height: 600px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 70%);
    top: -200px; right: -200px;
    pointer-events: none;
    z-index: 0;
    animation: drift 8s ease-in-out infinite alternate;
  }
  .glow-orb2 {
    position: fixed;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    bottom: -100px; left: -100px;
    pointer-events: none;
    z-index: 0;
    animation: drift 10s ease-in-out infinite alternate-reverse;
  }
  @keyframes drift {
    from { transform: translate(0,0); }
    to   { transform: translate(30px, 40px); }
  }

  .container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    position: relative;
    z-index: 1;
  }

  /* Header */
  header {
    text-align: center;
    padding: 3rem 0 2rem;
    animation: fadeUp 0.8s ease both;
  }
  .badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    letter-spacing: 0.15em;
    margin-bottom: 1.2rem;
    animation: pulse 3s ease infinite;
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,229,255,0.3); }
    50% { box-shadow: 0 0 0 8px rgba(0,229,255,0); }
  }
  h1 {
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
  }
  h1 span {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle {
    color: var(--muted);
    font-size: 1rem;
    max-width: 500px;
    margin: 0 auto 2rem;
    line-height: 1.6;
  }
  .header-links {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.6rem 1.3rem;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    text-decoration: none;
    transition: all 0.2s;
    cursor: pointer;
    border: none;
  }
  .btn-primary {
    background: var(--accent);
    color: #000;
  }
  .btn-primary:hover { background: #33ebff; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,229,255,0.4); }
  .btn-outline {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent); transform: translateY(-1px); }

  /* Stats row */
  .stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 2.5rem 0;
    animation: fadeUp 0.8s 0.2s ease both;
  }
  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .stat-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    display: block;
  }
  .stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem;
  }

  /* Upload section */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.8s 0.3s ease both;
  }
  .card-title {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: rgba(0,229,255,0.04);
  }
  .drop-zone input[type=file] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    width: 100%;
    height: 100%;
  }
  .drop-icon { font-size: 2.5rem; margin-bottom: 0.8rem; display: block; }
  .drop-text { color: var(--muted); font-size: 0.9rem; }
  .drop-text strong { color: var(--text); }
  .file-selected {
    margin-top: 1rem;
    padding: 0.7rem 1rem;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
    display: none;
  }

  .submit-btn {
    width: 100%;
    margin-top: 1.2rem;
    padding: 1rem;
    font-size: 0.9rem;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent2), #4f46e5);
    color: white;
    border: none;
    cursor: pointer;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.05em;
    transition: all 0.2s;
  }
  .submit-btn:hover:not(:disabled) { opacity: 0.85; transform: translateY(-1px); box-shadow: 0 6px 24px rgba(124,58,237,0.4); }
  .submit-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Loading */
  .loading {
    display: none;
    text-align: center;
    padding: 1.5rem;
  }
  .wave {
    display: inline-flex;
    gap: 5px;
    align-items: flex-end;
    height: 30px;
  }
  .wave span {
    display: block;
    width: 5px;
    background: var(--accent);
    border-radius: 3px;
    animation: wave 1s ease infinite;
  }
  .wave span:nth-child(1) { animation-delay: 0s; }
  .wave span:nth-child(2) { animation-delay: 0.1s; }
  .wave span:nth-child(3) { animation-delay: 0.2s; }
  .wave span:nth-child(4) { animation-delay: 0.3s; }
  .wave span:nth-child(5) { animation-delay: 0.4s; }
  @keyframes wave {
    0%,100% { height: 6px; }
    50% { height: 28px; }
  }
  .loading-text {
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-top: 0.8rem;
  }

  /* Result */
  #result { display: none; animation: fadeUp 0.5s ease both; }
  .result-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
  }
  .result-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
  }
  .result-badge.success { background: rgba(0,230,118,0.15); color: var(--success); }
  .result-badge.error   { background: rgba(255,82,82,0.15);  color: var(--error); }

  .result-meta {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.2rem;
  }
  .meta-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
  }
  .meta-val { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; color: var(--accent); }
  .meta-key { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

  /* MFCC Heatmap */
  .heatmap-wrap { overflow-x: auto; }
  canvas#heatmap {
    border-radius: 8px;
    width: 100%;
    image-rendering: pixelated;
  }
  .heatmap-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-top: 0.4rem;
  }

  /* Error message */
  .error-msg {
    color: var(--error);
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    background: rgba(255,82,82,0.08);
    border: 1px solid rgba(255,82,82,0.2);
    border-radius: 8px;
    padding: 1rem;
  }

  /* Architecture */
  .arch-grid {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .arch-layer {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.7rem 1rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    transition: border-color 0.2s;
  }
  .arch-layer:hover { border-color: var(--accent2); }
  .arch-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .arch-arrow { text-align: center; color: var(--muted); font-size: 0.8rem; }
  .arch-name { flex: 1; }
  .arch-shape { color: var(--muted); font-size: 0.72rem; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 600px) {
    .stats { grid-template-columns: 1fr 1fr; }
    .result-meta { grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>
<div class="glow-orb"></div>
<div class="glow-orb2"></div>

<div class="container">
  <header>
    <div class="badge">⚡ LIVE · LAB ASSIGNMENT 5</div>
    <h1>LSTM <span>Audio</span><br>Sequence Predictor</h1>
    <p class="subtitle">Upload an audio clip and watch the LSTM predict the next 5 MFCC frames using deep sequence learning.</p>
    <div class="header-links">
      <a href="/docs" class="btn btn-primary">📖 Swagger Docs</a>
      <a href="/health" class="btn btn-outline">💚 Health Check</a>
    </div>
  </header>

  <div class="stats">
    <div class="stat">
      <span class="stat-value">LSTM</span>
      <span class="stat-label">Model Type</span>
    </div>
    <div class="stat">
      <span class="stat-value">13</span>
      <span class="stat-label">MFCC Features</span>
    </div>
    <div class="stat">
      <span class="stat-value">20→5</span>
      <span class="stat-label">Frames In→Out</span>
    </div>
  </div>

  <!-- Upload Card -->
  <div class="card">
    <div class="card-title">🎙️ Try the Model</div>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="audioFile" accept=".wav,.mp3,.ogg,.flac,.m4a"/>
      <span class="drop-icon">🎵</span>
      <div class="drop-text"><strong>Drop an audio file here</strong><br>or click to browse · WAV, MP3, OGG supported</div>
    </div>
    <div class="file-selected" id="fileSelected"></div>
    <button class="submit-btn" id="submitBtn" disabled onclick="predict()">
      Run Prediction →
    </button>

    <div class="loading" id="loading">
      <div class="wave">
        <span></span><span></span><span></span><span></span><span></span>
      </div>
      <div class="loading-text">Extracting MFCCs & running LSTM inference...</div>
    </div>
  </div>

  <!-- Result Card -->
  <div class="card" id="result">
    <div class="result-header">
      <div class="card-title" style="margin:0">📊 Prediction Result</div>
      <span class="result-badge success" id="resultBadge">SUCCESS</span>
    </div>
    <div id="resultContent"></div>
  </div>

  <!-- Architecture Card -->
  <div class="card" style="animation-delay:0.5s">
    <div class="card-title">🧠 Model Architecture</div>
    <div class="arch-grid">
      <div class="arch-layer">
        <div class="arch-dot" style="background:#00e5ff"></div>
        <span class="arch-name">Input</span>
        <span class="arch-shape">(20, 13) — 20 frames × 13 MFCCs</span>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-layer">
        <div class="arch-dot" style="background:#7c3aed"></div>
        <span class="arch-name">LSTM(128) + BatchNorm</span>
        <span class="arch-shape">return_sequences=True</span>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-layer">
        <div class="arch-dot" style="background:#7c3aed"></div>
        <span class="arch-name">LSTM(64) + BatchNorm</span>
        <span class="arch-shape">return_sequences=False</span>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-layer">
        <div class="arch-dot" style="background:#f59e0b"></div>
        <span class="arch-name">Dense(128, relu) + Dropout(0.3)</span>
        <span class="arch-shape">hidden layer</span>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-layer">
        <div class="arch-dot" style="background:#00e676"></div>
        <span class="arch-name">Dense(65, linear)</span>
        <span class="arch-shape">5 frames × 13 MFCCs = 65 outputs</span>
      </div>
    </div>
  </div>
</div>

<script>
const fileInput = document.getElementById('audioFile');
const dropZone = document.getElementById('dropZone');
const fileSelected = document.getElementById('fileSelected');
const submitBtn = document.getElementById('submitBtn');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('result');
const resultBadge = document.getElementById('resultBadge');
const resultContent = document.getElementById('resultContent');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    fileSelected.style.display = 'block';
    fileSelected.textContent = '✓ ' + fileInput.files[0].name + '  (' + (fileInput.files[0].size/1024).toFixed(1) + ' KB)';
    submitBtn.disabled = false;
  }
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; fileInput.dispatchEvent(new Event('change')); }
});

async function predict() {
  const file = fileInput.files[0];
  if (!file) return;

  submitBtn.disabled = true;
  loading.style.display = 'block';
  resultCard.style.display = 'none';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();
    loading.style.display = 'none';
    resultCard.style.display = 'block';

    if (!res.ok) {
      resultBadge.className = 'result-badge error';
      resultBadge.textContent = 'ERROR ' + res.status;
      resultContent.innerHTML = '<div class="error-msg">⚠️ ' + (data.detail || JSON.stringify(data)) + '</div>';
      return;
    }

    resultBadge.className = 'result-badge success';
    resultBadge.textContent = 'SUCCESS';

    const frames = data.predicted_frames;
    resultContent.innerHTML = `
      <div class="result-meta">
        <div class="meta-item"><div class="meta-val">${data.input_duration_sec}s</div><div class="meta-key">Input Duration</div></div>
        <div class="meta-item"><div class="meta-val">${data.pred_len}</div><div class="meta-key">Frames Predicted</div></div>
        <div class="meta-item"><div class="meta-val">${data.n_mfcc}</div><div class="meta-key">MFCC Coefficients</div></div>
      </div>
      <div style="margin-bottom:0.6rem;font-size:0.8rem;color:var(--muted);">Predicted MFCC Heatmap (${data.pred_len} frames × ${data.n_mfcc} coefficients)</div>
      <div class="heatmap-wrap"><canvas id="heatmap" height="100"></canvas></div>
      <div class="heatmap-label"><span>Frame 1</span><span>Frame ${data.pred_len}</span></div>
    `;

    drawHeatmap(frames);

  } catch(e) {
    loading.style.display = 'none';
    resultCard.style.display = 'block';
    resultBadge.className = 'result-badge error';
    resultBadge.textContent = 'ERROR';
    resultContent.innerHTML = '<div class="error-msg">⚠️ Network error: ' + e.message + '</div>';
  }

  submitBtn.disabled = false;
}

function drawHeatmap(frames) {
  const canvas = document.getElementById('heatmap');
  if (!canvas) return;
  const rows = frames[0].length; // n_mfcc
  const cols = frames.length;    // pred_len
  canvas.width = cols * 60;
  canvas.height = rows * 20;
  const ctx = canvas.getContext('2d');

  const flat = frames.flat();
  const min = Math.min(...flat), max = Math.max(...flat);

  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const v = (frames[c][r] - min) / (max - min + 1e-9);
      ctx.fillStyle = heatColor(v);
      ctx.fillRect(c * 60, r * 20, 59, 19);
    }
  }
}

function heatColor(t) {
  // blue → cyan → green → yellow → red
  const stops = [
    [0,   [8,12,30]],
    [0.25,[[0,100,180]]],
    [0.5, [[0,200,200]]],
    [0.75,[[0,220,120]]],
    [1,   [[255,100,50]]]
  ];
  let lo = stops[0], hi = stops[stops.length-1];
  for (let i = 0; i < stops.length-1; i++) {
    if (t >= stops[i][0] && t <= stops[i+1][0]) { lo = stops[i]; hi = stops[i+1]; break; }
  }
  const f = (t - lo[0]) / (hi[0] - lo[0] + 1e-9);
  const lerp = (a,b) => Math.round(a + f*(b-a));
  const [r,g,b] = lo[1][0].map ? lo[1][0].map((v,i)=>lerp(v,hi[1][0][i])) : lo[1].map((v,i)=>lerp(v,hi[1][i]));
  return `rgb(${r},${g},${b})`;
}
</script>
</body>
</html>"""

# ── Routes ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=HOMEPAGE)

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
