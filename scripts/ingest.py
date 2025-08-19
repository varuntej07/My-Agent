import os, glob, hashlib, datetime, json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from pypdf import PdfReader

CHROMA_DIR = "storage/chroma"            # where Chroma stores its index files
COLLECTION = "varun_kb"
CHUNK_SIZE = 800                         # how big each text chunk is (characters)
CHUNK_OVERLAP = 120                      # how much chunks overlap to avoid mid-sentence cuts
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"    # Embedding model name from SentenceTransformers

DATA_DIR = Path("data/raw")              # all the knowledge files to ingest are here


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

def load_md_or_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text

# Chunking: splits long text into overlapping windows
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+size]
        if not chunk:
            break
        chunks.append(chunk)
        i += max(size - overlap, 1)
    return chunks

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
    elif ext in (".md", ".txt"):
        meta["source"] = "markdown" if ext == ".md" else "text"
        text = load_md_or_txt(path)
    elif ext in (".html", ".htm"):
        meta["source"] = "html"
        text = load_html(path)
    elif ext == ".json" and "ig" in path.stem.lower():
        meta["source"] = "instagram"
        text = load_instagram_json(path)
    else:
        text = ""

    return text, meta


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
        print("No files found in data/raw. Add resume.pdf / skills.md first.")
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



if __name__ == "__main__":
    main()
