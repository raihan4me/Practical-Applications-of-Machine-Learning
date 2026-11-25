FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

# Same pinned versions as training: the pickled model requires scikit-learn 1.4.2
COPY serve/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY serve/ ./serve/
COPY models/ ./models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
