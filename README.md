# 🎵 LSTM Audio Sequence Prediction — FastAPI on Render

> Lab Assignment 5 | Audio Prediction | Deployment: FastAPI + Render.com

---

## Dataset Declaration

| Field | Details |
|-------|---------|
| **Name** | Mozilla Common Voice 11.0 (English) |
| **Source** | https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0 |
| **Description** | Crowd-sourced speech recordings. 13 MFCC features extracted per audio frame. |
| **Preprocessing** | Resample → 16kHz → Extract MFCCs → MinMaxScaler normalize → Sliding window (20 in, 5 out) |

---

## Project Structure

```
lstm-audio-prediction/
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── LSTM_Audio_Prediction.ipynb # Colab training notebook
├── saved_model/                # Generated after training
│   ├── lstm_audio_model.keras
│   ├── scaler.pkl
│   └── config.json
└── README.md
```

---

## LSTM Mathematical Model

**Forget gate** — what to erase from memory:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Input gate** — what new info to add:
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

**Cell state update** (long-term memory):
```
C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ C̃_t
```

**Output gate** — what to expose as output:
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

---

## Model Architecture

```
Input  (20, 13)  ← 20 MFCC frames × 13 coefficients
  ↓
LSTM(128, return_sequences=True) + BatchNorm
  ↓
LSTM(64) + BatchNorm
  ↓
Dense(128, relu) + Dropout(0.3)
  ↓
Dense(65, linear)  ← predicts 5 frames × 13 MFCCs
```

---

## API Reference

### GET /health
```json
{ "status": "ok", "model_ready": true }
```

### POST /predict
- **Input:** multipart audio file (WAV / MP3 / OGG)
- **Output:**
```json
{
  "predicted_frames": [[...13 values...], ...5 frames],
  "pred_len": 5,
  "n_mfcc": 13,
  "input_duration_sec": 2.048,
  "message": "Predicted 5 future MFCC frames from 2.05s of audio."
}
```

**Live API:** `https://YOUR-APP-NAME.onrender.com`  
**Swagger docs:** `https://YOUR-APP-NAME.onrender.com/docs`

---

## Deployment

Deployed on **Render.com** (free tier) using `render.yaml`.  
Auto-deploys on every push to the `main` branch.

---

## Submission

- **Colab Link:** ___________________________
- **GitHub Repo:** ___________________________
- **Live API URL:** ___________________________

---

## Team Contributions

| Member | Contribution |
|--------|-------------|
| Member 1 | Dataset collection, MFCC preprocessing |
| Member 2 | LSTM model design and training |
| Member 3 | FastAPI deployment, Render setup |
| Member 4 | Testing, documentation, GitHub |

---

## AI Tool Acknowledgement

| Tool | Purpose | Sections |
|------|---------|---------|
| Claude (Anthropic) | Code scaffolding, README template | Notebook structure, API boilerplate, README |
