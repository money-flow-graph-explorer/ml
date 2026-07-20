FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py train.py ./
# model.json + feature_names.json are provided at runtime (mounted volume or baked after training).

ENV MODEL_DIR=/app
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
