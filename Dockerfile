FROM python:3.11-slim

WORKDIR /app
# Install dependencies first so Docker caches this layer

# (use a slim API-only file, see note below; requirements.txt also works)

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app, prediction API module, and model artifacts.

COPY src/app.py ./src/app.py
COPY src/api/ ./src/api/

COPY models/ ./models/

# Run as a non-root user

RUN useradd --create-home appuser && chown -R appuser /app

USER appuser

EXPOSE 8000

# Streamlit exposes this health endpoint.

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \

    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/_stcore/health')" || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.port=8000", "--server.address=0.0.0.0"]