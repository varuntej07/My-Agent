import os, glob, hashlib, datetime, json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document

CHROMA_DIR = "storage/chroma"            # where Chroma stores its index files
COLLECTION = "varun_kb"
CHUNK_SIZE = 800                         # sized to fit one structured block per chunk
CHUNK_OVERLAP = 150                      # overlap to capture block boundaries
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"    # Embedding model name from SentenceTransformers

# DATA_DIR = Path("data/Ooink")
DATA_DIR = Path("data/my raw data")              # all the knowledge files to ingest are here
EMBEDDINGS_JSON_PATH = "storage/embeddings.json"  # Pre-computed embeddings export


# Embedding wrapper: makes MiniLM look like a Chroma EmbeddingFunction
class MiniLMEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.model(texts)

# converts file content to plain text we can embed and store
def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return "\n".join(pages)

def load_docx(path: Path) -> str:
    doc = Document(str(path))
    paragraphs = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)
    return "\n".join(paragraphs)

def load_md_or_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_json_kb(path: Path) -> str:
    """
    Load and flatten a knowledge base JSON file into readable text for embedding. 
    Preserves structure but makes it text-friendly.
    
    Handles nested dicts, lists, and produces natural language output.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {path}: {e}")
        return ""
    
    text_parts = []
    
    def flatten_value(key: str, value, depth: int = 0) -> str:
        """Recursively convert any JSON value to readable text"""
        indent = "  " * depth
        
        if isinstance(value, dict):
            # For dicts, show key-value pairs
            lines = [f"{key}:"]
            for k, v in value.items():
                lines.append(flatten_value(k, v, depth + 1))
            return "\n".join(lines)
        
        elif isinstance(value, list):
            # For lists, show items
            lines = [f"{key}:"]
            for i, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    lines.append(flatten_value(f"Item {i+1}", item, depth + 1))
                else:
                    lines.append(f"{indent}  - {item}")
            return "\n".join(lines)
        
        elif isinstance(value, bool):
            return f"{indent}{key}: {str(value)}"
        
        elif value is None:
            return f"{indent}{key}: (not specified)"
        
        else:
            # Strings, numbers, etc.
            return f"{indent}{key}: {value}"
    
    # Top-level structure
    for key, value in data.items():
        text_parts.append(flatten_value(key, value))
    
    return "\n".join(text_parts)

# Chunking: splits text into chunks that respect semantic block boundaries.
# Blocks are separated by double newlines. Small blocks are merged until they
# would exceed the size limit. Large blocks are split with overlap.
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Split on double-newline to get logical blocks (projects, roles, etc.)
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    chunks = []
    current = ""

    for block in blocks:
        # Normalize whitespace within a block but preserve block boundaries
        block = " ".join(block.split())

        if not current:
            current = block
        elif len(current) + len(block) + 1 <= size:
            # Merge small adjacent blocks
            current = current + " " + block
        else:
            # Flush current chunk
            chunks.append(current)
            current = block

    if current:
        chunks.append(current)

    # Split any oversized chunks using character-based fallback
    final = []
    for chunk in chunks:
        if len(chunk) <= size:
            final.append(chunk)
        else:
            i = 0
            while i < len(chunk):
                piece = chunk[i:i + size]
                if piece:
                    final.append(piece)
                i += max(size - overlap, 1)

    return final

# Helpers: hashing for dedup; metadata for traceability
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def file_metadata(path: Path, source_type: str) -> dict:
    stat = path.stat()
    modified = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    return {
        "source": source_type,
        "path": str(path),
        "filename": path.name,
        "modified": modified,
    }

def load_file(path: Path) -> tuple[str, dict]:
    ext = path.suffix.lower()
    meta = file_metadata(path, "unknown")

    if ext == ".pdf":
        meta["source"] = "resume" if "resume" in path.stem.lower() else "pdf"
        text = load_pdf(path)
    elif ext == ".docx":
        meta["source"] = "resume" if "resume" in path.stem.lower() else "docx"
        text = load_docx(path)
    elif ext in (".md", ".txt"):
        meta["source"] = "markdown" if ext == ".md" else "text"
        text = load_md_or_txt(path)
    elif ext == ".json":
        # Handle ANY JSON file
        if "menu" in path.stem.lower() or "kb" in path.stem.lower():
            meta["source"] = "knowledge_base"
        elif "ig" in path.stem.lower():
            meta["source"] = "instagram"
        else:
            meta["source"] = "json_data"

        # Use intelligent JSON loader
        text = load_json_kb(path)
    else:
        text = ""

    return text, meta

def export_embeddings_to_json(collection, output_path: str):
    """Export all chunks and their embeddings from ChromaDB to JSON file"""
    try:
        # Get all documents from the collection
        result = collection.get(include=['documents', 'embeddings', 'metadatas'])
        
        if not result['ids']:
            print("No documents found in collection to export")
            return
            
        # Build the JSON structure
        export_data = {
            "model_name": EMBED_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "exported_at": datetime.datetime.now().isoformat(),
            "chunks": []
        }
        
        for i, chunk_id in enumerate(result['ids']):
            # Convert numpy array to list for JSON serialization
            embedding = result['embeddings'][i]
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            chunk_data = {
                "id": chunk_id,
                "text": result['documents'][i],
                "embedding": embedding,
                "metadata": result['metadatas'][i] if result['metadatas'] else {}
            }
            export_data["chunks"].append(chunk_data)
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"Exported {len(export_data['chunks'])} chunks with embeddings to {output_path}")
        print(f"Model: {EMBED_MODEL_NAME}")
        print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"Failed to export embeddings: {e}")

def main():
    # 1) Open/Create a persistent Chroma database in CHROMA_DIR
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedder = MiniLMEmbeddingFunction(EMBED_MODEL_NAME)

    # Try to fetch the existing collection; if missing, create one bound to our embedder
    try:
        coll = client.get_collection(COLLECTION)
    except Exception:
        coll = client.create_collection(name=COLLECTION, embedding_function=embedder)

    # If the collection existed without our embedder, recreate it once:
    if coll._embedding_function is None:
        client.delete_collection(COLLECTION)
        coll = client.create_collection(name=COLLECTION, embedding_function=embedder)

    # 2) find files to ingest in data/raw
    patterns = [
        str(DATA_DIR / "*.pdf"),
        str(DATA_DIR / "*.docx"),
        str(DATA_DIR / "*.md"),
        str(DATA_DIR / "*.txt"),
        str(DATA_DIR / "*.html"),
        str(DATA_DIR / "*.htm"),
        str(DATA_DIR / "*.json"),
        str(DATA_DIR / "*" / "*.*"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    # Dedup, sort, keep only actual files
    files = [Path(p) for p in sorted(set(files)) if Path(p).is_file()]

    if not files:
        print(f"No files found in {DATA_DIR}. Add files to embed...")
        return

    print(f"Found {len(files)} files")

    # 3) Build lists of chunk IDs, texts, and metadata to upsert in one shot
    add_ids, add_docs, add_metas = [], [], []

    for f in files:
        text, meta = load_file(f)
        if not text.strip():
            print(f"Skipping (no text): {f}")
            continue

        # Hashing the entire file's text so we can make deterministic chunk IDs
        content_hash = sha256(text)
        meta["content_hash"] = content_hash

        # Split into overlapping chunks
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        for idx, ch in enumerate(chunks):
            cid = f"{content_hash[:16]}_{idx}"               # Unique id per chunk: first 16 chars of hash + chunk index
            add_ids.append(cid)
            add_docs.append(ch)
            add_metas.append({**meta, "chunk_index": idx})

        print(f"Ingested {f.name}: {len(chunks)} chunks")

    if not add_ids:
        print("Nothing to add.")
        return

    # 4) Write everything into Chroma. Re-running with identical content just overwrites the same ids (no duplication).
    coll.upsert(ids=add_ids, documents=add_docs, metadatas=add_metas)
    print(f"Upserted {len(add_ids)} chunks into collection '{COLLECTION}'")
    
    # 5) Auto-export embeddings to JSON for fast server startup
    print("\n Auto-exporting embeddings to JSON...")
    export_embeddings_to_json(coll, EMBEDDINGS_JSON_PATH)
    
    
if __name__ == "__main__":
    main()