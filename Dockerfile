FROM python:3.11-slim

WORKDIR /app
# Install dependencies first so Docker caches this layer

# (use a slim API-only file, see note below; requirements.txt also works)

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs

COPY src/api/ ./src/api/

COPY models/ ./models/

# Run as a non-root user

RUN useradd --create-home appuser && chown -R appuser /app

USER appuser

EXPOSE 8000

# Marks the container unhealthy if /health stops responding

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \

    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# --app-dir lets main.py keep `from api import ...` and `from schemas import ...`

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]