from fastapi import FastAPI, File, UploadFile, HTTPException, Response
import numpy as np
import io
import time
import logging
from PIL import Image
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("cats-dogs-api")

app = FastAPI(title="Cats vs Dogs Classifier API")

MODEL_PATH = "cats_dogs_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Cat", "Dog"]

model = None

# --- Prometheus metrics ---
# Counter: only ever goes up, good for "how many total requests"
PREDICTION_COUNTER = Counter(
    "prediction_requests_total", "Total number of prediction requests", ["prediction"]
)
# Histogram: buckets response times, good for "how fast are we, typically"
LATENCY_HISTOGRAM = Histogram(
    "prediction_request_latency_seconds", "Time taken to process a prediction request"
)


@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(image), axis=0)
    return img_array


def interpret_prediction(raw_score: float) -> dict:
    predicted_class = CLASS_NAMES[int(raw_score > 0.5)]
    confidence = raw_score if raw_score > 0.5 else 1 - raw_score
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "raw_score": round(raw_score, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "version": "1.1"}


@app.get("/metrics")
def metrics():
    # Standard Prometheus scrape endpoint - returns plain text in a
    # specific format Prometheus/Grafana know how to parse.
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start_time = time.time()

    contents = await file.read()
    try:
        img_array = preprocess_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    result = interpret_prediction(raw_score)

    latency = time.time() - start_time
    LATENCY_HISTOGRAM.observe(latency)
    PREDICTION_COUNTER.labels(prediction=result["prediction"]).inc()

    # We log the filename and result, but deliberately never log the
    # actual image bytes themselves - no reason to store that.
    logger.info(
        f"Prediction request | filename={file.filename} | "
        f"prediction={result['prediction']} | confidence={result['confidence']} | "
        f"latency={latency:.4f}s"
    )

    return result
