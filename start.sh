#!/bin/bash
set -euo pipefail

# Ensure the Chroma path exists (baked-in index will sit here)
mkdir -p /app/storage/chroma

# Start Ollama in the background
ollama serve &

# Give Ollama a moment to come up
sleep 2

# Start Flask via Gunicorn, binding to the Cloud Run port
exec gunicorn --bind 0.0.0.0:${PORT:-8080} --timeout 300 --workers 1 server:app
