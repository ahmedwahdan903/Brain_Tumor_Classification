from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
from PIL import Image
import uvicorn
import os

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = r"E:\ahmed\did it\brain_tumor_model.keras"   # ← بعد التحويل
INPUT_SHAPE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6
CLASS_NAMES = ["glioma", "notumor", "meningioma", "pituitary"]  # ← غيّرها لو مختلفة

# ─── Lifespan ────────────────────────────────────────────────────────────────
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  '{MODEL_PATH}' not found — run convert_model.py first!")
    else:
        tf.keras.config.enable_unsafe_deserialization()
        model = load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded from {MODEL_PATH}")
    yield

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Detection API",
    description="EfficientNetB3-based Brain Tumor Classification Model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    confidence_pct: str
    all_probabilities: dict[str, float]
    low_confidence: bool
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize(INPUT_SHAPE)
    img_array = np.array(image, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3) في [0,255]

# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {"message": "Brain Tumor Detection API 🧠 — visit /docs"}


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok" if model else "model_not_loaded",
        model_loaded=model is not None,
        classes=CLASS_NAMES
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload JPG or PNG only.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        img_array = preprocess_image(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image error: {e}")

    preds = model.predict(img_array, verbose=0)[0]
    pred_index = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_index]
    pred_conf = float(preds[pred_index])
    low_conf = pred_conf < CONFIDENCE_THRESHOLD

    return PredictionResponse(
        predicted_class=pred_class,
        confidence=round(pred_conf, 4),
        confidence_pct=f"{pred_conf*100:.2f}%",
        all_probabilities={c: round(float(p), 4) for c, p in zip(CLASS_NAMES, preds)},
        low_confidence=low_conf,
        message=(
            "Low confidence — upload a clearer MRI image." if low_conf
            else f"Detected: {pred_class} with {pred_conf*100:.1f}% confidence."
        )
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)