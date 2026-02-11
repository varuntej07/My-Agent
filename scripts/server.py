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

SYSTEM_PROMPT = """You are an AI assistant that answers questions about Varun Tej — a software engineer with a Master's in Computer Science from Seattle University.

Your job is to provide clear, accurate, and professional responses based ONLY on the provided context.

Rules:
1. Answer in third person (say "Varun" or "he", never "I" or "me").
2. Be concise and factual. Do not add filler, jokes, or unnecessary commentary.
3. Stick strictly to the provided context. If the context does not contain enough information to answer the question, respond with: "I don't have enough information about that yet. Varun is still adding more data to this agent."
4. Never hallucinate or invent details not present in the context.
5. Never reveal private information like phone numbers, addresses, or API keys.
6. If the question is completely unrelated to Varun, politely say: "I'm Varun's AI assistant and can only answer questions about him. Feel free to ask about his work, skills, or experience."
7. Do not use citations, markdown headers, or bullet points unless the answer genuinely benefits from a list format.
8. Keep responses under 150 words unless the question requires more detail."""

def build_prompt(query: str, context_chunks: list[str]) -> tuple[str, str]:
    """Returns (system_message, user_message) tuple."""
    if not context_chunks:
        system = SYSTEM_PROMPT
        user = f"Question: {query}\n\nContext: No relevant context was found."
        return system, user

    context = "\n\n---\n\n".join(context_chunks)
    user = f"Question: {query}\n\nContext:\n{context}"
    return SYSTEM_PROMPT, user

def _answer_with_openai(system_msg: str, user_msg: str) -> str:
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
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.3,
                "max_tokens": MAX_OUT,
            },
            timeout=90
        )

        print(f"OpenAI status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"].strip()
                print(f"Response: {content[:100]}...")
                return content

        print(f"OpenAI error - Status: {response.status_code}, Response: {response.text[:200]}")
        return ""

    except Exception as e:
        print(f"OpenAI request failed: {e}")
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
        
        system_msg, user_msg = build_prompt(query, ctx)
        content = _answer_with_openai(system_msg, user_msg)

        if not content:
            content = "Sorry, I couldn't generate a response right now. Please try again."
        
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
