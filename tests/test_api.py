import sys
import os
import io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
from main import preprocess_image, interpret_prediction


def make_fake_image_bytes(size=(300, 300), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_preprocess_image_shape_and_size():
    fake_bytes = make_fake_image_bytes()
    result = preprocess_image(fake_bytes)
    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.uint8


def test_interpret_prediction_dog():
    result = interpret_prediction(0.9)
    assert result["prediction"] == "Dog"
    assert result["confidence"] == 0.9


def test_interpret_prediction_cat():
    result = interpret_prediction(0.1)
    assert result["prediction"] == "Cat"
    assert result["confidence"] == 0.9


def test_interpret_prediction_boundary():
    result = interpret_prediction(0.5)
    assert result["prediction"] in ["Cat", "Dog"]
