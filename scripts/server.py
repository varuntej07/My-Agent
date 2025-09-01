import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# absolute path for embeddings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_JSON_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "storage", "embeddings.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
MAX_OUT = 400

app = Flask(__name__)
CORS(app, origins=[
    "https://api.varuntej.dev",
    "https://varuntej.dev", 
    "http://localhost:3000",
    "http://localhost:8501"
])

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global variables for pre-loaded embeddings
embeddings_data = None
embedding_model = None

def load_embeddings():
    """Load pre-computed embeddings from JSON file"""
    global embeddings_data, embedding_model
    
    if not os.path.exists(EMBEDDINGS_JSON_PATH):
        print(f"Embeddings file not found: {EMBEDDINGS_JSON_PATH}")
        print("Run 'python scripts/ingest.py' first to generate embeddings.json")
        return False
    
    try:
        with open(EMBEDDINGS_JSON_PATH, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)

            # L2-normalize all stored embeddings once at load time
            for ch in embeddings_data.get('chunks', []):
                v = np.asarray(ch['embedding'], dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                ch['embedding'] = v.tolist()  # overwrite with normalized vector
        
        # Load the embedding model for query encoding
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        chunks_count = len(embeddings_data.get('chunks', []))
        model_name = embeddings_data.get('model_name', 'unknown')
        file_size = os.path.getsize(EMBEDDINGS_JSON_PATH) / 1024
        
        print(f"Loaded {chunks_count} pre-computed embeddings")
        print(f"Model: {model_name}")
        print(f"File size: {file_size:.1f} KB")
        print(f"Path: {EMBEDDINGS_JSON_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return False

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_context(query: str) -> list[str]:
    """Retrieve top-k most similar chunks using pre-computed embeddings"""
    global embeddings_data, embedding_model
    
    if not embeddings_data or not embedding_model:
        print("❌ Embeddings not loaded")
        return []
    
    try:
        # Embed the query using the same model
        query_embedding = embedding_model.encode([query])[0].astype(np.float32)
        
        # ensures the query is also a unit vector, so cosine becomes a pure dot product
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding = query_embedding / q_norm
        
        # Calculate similarities with all chunks
        similarities = []
        for chunk in embeddings_data['chunks']:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((similarity, chunk['text']))
        
        # Sort by similarity (highest first) and return top-k texts
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [text for _, text in similarities[:TOP_K]]
        
        return top_chunks
        
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        return []

def build_prompt(query: str, context_chunks: list[str]) -> str:
    if not context_chunks:
        return f"""You are Varun's personal clone, but you must respond exactly with this message:

"Varun is still improving the model by injecting more data. This is in active development, so it may not know everything yet."

User question: {query}"""
    
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are Varun's personal clone. 
            Style: direct, confident, witty when appropriate.
            Always stay in character as Varun's AI twin.      

            Rules:
            - Prefer facts from the provided context.
            - If info is missing, unclear, or weak: Explicitly say "Varun is still improving the model by injecting more data. This is in active development, so it may not know everything yet."
            - Never hallucinate or make up unverifiable claims.  
            - If the question is not about Varun, politely throw a hilarious joke and divert the topic.
            - Never reveal private/PII.  
            - Never use citations in the response.

            User question: {query}

            Context (snippets about Varun):
            {context}
            """

def _answer_with_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": MAX_OUT,
            },
            timeout=90
        )
        
        print(f"OpenAI status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"].strip()
                print(f"Varun' clone replied: {content[:100]}...")
                return content
        
        print(f"Varun' clone failed to answer - Status: {response.status_code}, Response: {response.text[:200]}")
        return ""
        
    except Exception as e:
        return ""

@app.route("/health", methods=["GET"])
def health():
    embeddings_loaded = embeddings_data is not None
    return jsonify({
        "status": "ok",
        "embeddings_loaded": embeddings_loaded,
        "chunks_count": len(embeddings_data.get('chunks', [])) if embeddings_data else 0
    }), 200

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}
        query = (data.get("query") or "").strip()

        if not query:
            return jsonify({
                "error": "Missing 'query' in request body.",
                "context_used": []
            }), 400
        
        ctx = retrieve_context(query)
        
        if not ctx and not embeddings_data:
            return jsonify({
                "error": "Knowledge base not initialized. Run 'python scripts/ingest.py' first.",
                "context_used": []
            }), 503
        
        prompt = build_prompt(query, ctx)
        content = _answer_with_openai(prompt)
        
        if not content:
            content = "WTF!!!!! why is there no content"
        
        return jsonify({"answer": content, "context_used": ctx})
        
    except Exception as e:
        return jsonify({
            "error": "Something went wrong while processing your request.",
            "context_used": []
        }), 503

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    
    # Load embeddings at startup
    print("🚀 Starting Varun's AI Clone Server...")
    
    if load_embeddings():
        print(f"Server ready on port {port}")
    else:
        print(f"Server starting on port {port} but embeddings not loaded")
        print("Some endpoints may not work until you run ingest.py")
    
    app.run(host='0.0.0.0', port=port, debug=False)
