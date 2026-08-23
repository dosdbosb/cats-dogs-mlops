import sys
import time
import io
import requests
from PIL import Image

BASE_URL = "http://localhost:8000"


def make_test_image_bytes():
    img = Image.new("RGB", (300, 300), (100, 150, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def check_health():
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "ok", f"Unexpected health response: {data}"
    assert data.get("model_loaded") is True, "Model did not load"
    print("Health check passed:", data)


def check_predict():
    files = {"file": ("test.jpg", make_test_image_bytes(), "image/jpeg")}
    resp = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    assert data.get("prediction") in ["Cat", "Dog"], f"Unexpected prediction: {data}"
    print("Predict check passed:", data)


if __name__ == "__main__":
    # Give the freshly-restarted container a few seconds to fully start
    # before hammering it with requests.
    time.sleep(5)
    try:
        check_health()
        check_predict()
        print("\nAll smoke tests passed.")
    except Exception as e:
        print(f"\nSmoke test FAILED: {e}")
        sys.exit(1)