from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import io
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Cats vs Dogs Classifier API")

MODEL_PATH = "cats_dogs_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Cat", "Dog"]

model = None

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into a model-ready array.
    Pulled out as its own function so it can be unit tested
    without needing a running server or a loaded model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(image), axis=0)
    return img_array


def interpret_prediction(raw_score: float) -> dict:
    """Turn a raw sigmoid output (0-1) into a labeled result.
    Separated out so the decision-boundary logic can be tested
    with plain numbers, no model or image required."""
    predicted_class = CLASS_NAMES[int(raw_score > 0.5)]
    confidence = raw_score if raw_score > 0.5 else 1 - raw_score
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "raw_score": round(raw_score, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        img_array = preprocess_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    return interpret_prediction(raw_score)