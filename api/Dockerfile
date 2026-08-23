FROM python:3.12-slim

# TensorFlow's CPU math operations depend on this library, which the
# slim base image doesn't include by default.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copied and installed first, separately from the rest of the code,
# so Docker can reuse this slow step from cache on future rebuilds
# unless requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual app code and the trained model.
COPY main.py .
COPY cats_dogs_model.keras .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]