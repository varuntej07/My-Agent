FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

WORKDIR /app

# Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code & baked Chroma index (see step C)
COPY . .

# Fix line endings and permissions for start.sh
RUN sed -i 's/\r$//' start.sh && \
    chmod +x start.sh

# Pre-pull model into image (start server just for the pull)
# NOTE: keeps image large but avoids cold-start downloads
RUN apt-get update && apt-get install -y bash curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Entrypoint
EXPOSE 8080
CMD ["./start.sh"]
