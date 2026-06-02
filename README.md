# 🎵 LSTM Audio Sequence Predictor

>  Deep Learning | FastAPI Deployment | Render.com

[![Live API](https://img.shields.io/badge/API-Live-brightgreen)](https://lstm-audio-prediction.onrender.com)
[![Docs](https://img.shields.io/badge/Swagger-Docs-blue)](https://lstm-audio-prediction.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-yellow)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)](https://tensorflow.org)

---

## 📌 Links

| Resource | URL |
|----------|-----|
| 🌐 **Live API** | https://lstm-audio-prediction.onrender.com |
| 📖 **Swagger UI** | https://lstm-audio-prediction.onrender.com/docs |

---

## 📊 Dataset

| Field | Details |
|-------|---------|
| **Name** | UrbanSound8K (via Hugging Face) |
| **Source** | `danavery/urbansound8K` |
| **Description** | Urban sound recordings across 10 classes (sirens, engines, music, etc.) |
| **Samples Used** | 500 audio clips for training |
| **Preprocessing** | Resample → 16kHz → 13 MFCCs → MinMaxScaler → Sliding window (20 in, 5 out) |

---

## 🧠 Model Architecture

```
Input  (20, 13)  ← 20 MFCC frames × 13 coefficients
        ↓
  LSTM(128, return_sequences=True)
  BatchNormalization + Dropout(0.2)
        ↓
  LSTM(64)
  BatchNormalization + Dropout(0.2)
        ↓
  Dense(128, activation='relu')
  Dropout(0.3)
        ↓
  Dense(65, activation='linear')   ← predicts 5 frames × 13 MFCCs
```

### LSTM Gate Equations

| Gate | Formula |
|------|---------|
| **Forget** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ |
| **Input** | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ |
| **Cell** | $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ |
| **Output** | $h_t = o_t \odot \tanh(C_t)$ |

---

## 🚀 API Reference

### `GET /`
Returns status message and docs link.

### `GET /health`
```json
{ "status": "ok", "model_ready": true }
```

### `POST /predict`
Upload an audio file (WAV / MP3 / OGG) and receive predicted MFCC frames.

**Request:** `multipart/form-data` with a `file` field.

**Response:**
```json
{
  "predicted_frames": [[...13 values...], ...5 frames],
  "pred_len": 5,
  "n_mfcc": 13,
  "input_duration_sec": 2.048,
  "message": "Predicted 5 future MFCC frames from 2.05s of audio."
}
```

---

## 📁 Project Structure

```
lstm-audio-prediction/
├── main.py                      # FastAPI application
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render deployment config
├── .python-version              # Pins Python 3.11
├── LSTM_Audio_Prediction.ipynb  # Colab training notebook
├── saved_model/
│   ├── lstm_audio_model.keras   # Trained LSTM model
│   ├── scaler.pkl               # MinMaxScaler
│   └── config.json              # Audio processing config
└── README.md
```

---

## 🛠️ Local Setup

```bash
git clone https://github.com/your_username/lstm-audio-prediction
cd lstm-audio-prediction
pip install -r requirements.txt
uvicorn main:app --reload
# Visit http://localhost:8000/docs
```

---

## ☁️ Deployment

Deployed on **Render.com** free tier using `render.yaml`.
Auto-deploys on every push to `main` branch.


## 🤖 AI Tool Acknowledgement

| Tool | Purpose |
|------|---------|
| Claude (Anthropic) | Code scaffolding, debugging, README |
| Google Colab | Model training environment |
