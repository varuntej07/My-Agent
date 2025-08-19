# my-agent

A personal AI agent that can answer questions about my background, skills, and experiences using local document embeddings.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your documents to `data/raw/` directory (PDFs, Markdown, Text files supported)

3. Run the ingest script to process documents:
```bash
python scripts/ingest.py
```

## Usage

1. Start the backend server:
```bash
python scripts/server.py
```

2. Start the Streamlit frontend:
```bash
streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Features

- Document ingestion with text chunking and embeddings
- Vector similarity search using ChromaDB
- Simple web interface with Streamlit
- API backend for extensibility