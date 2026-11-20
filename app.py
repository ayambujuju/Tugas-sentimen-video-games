import json, time
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

BASE = Path(__file__).parent
INFO = json.loads((BASE / "models" / "model_info.json").read_text())
VERSION = INFO["active_version"]
model = joblib.load(BASE / INFO["models"][VERSION]["file"])
LOG_FILE = BASE / "predictions.log"

app = FastAPI(title="Sentimen Ulasan Video Games")

class Review(BaseModel):
    title: str = ""
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return INFO

@app.post("/predict")
def predict(review: Review):
    # format input HARUS sama dengan saat training: "judul teks"
    teks = f"{review.title} {review.text}".strip()
    proba = model.predict_proba([teks])[0]
    pred = int(proba.argmax())
    hasil = {
        "label": "positif" if pred == 1 else "negatif",
        "probability": round(float(proba[pred]), 4),
        "model_version": VERSION,
    }
    # observability: catat tiap prediksi
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({"waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "input": teks[:500], **hasil}) + "\n")
    return hasil
