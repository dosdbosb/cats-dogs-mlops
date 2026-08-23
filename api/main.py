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
    # Loaded once when the server starts, not on every request —
    # loading a ~42MB model file per request would be painfully slow.
    global model
    model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    image = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(image), axis=0)
    # Note: no manual normalization here — rescaling is a layer baked
    # into the model itself, so raw 0-255 pixel values go straight in.

    prediction = float(model.predict(img_array, verbose=0)[0][0])
    predicted_class = CLASS_NAMES[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "raw_score": round(prediction, 4),
    }