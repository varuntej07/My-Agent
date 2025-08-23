import os
from flask import Flask, request, jsonify
import chromadb
from flask_cors import CORS
import requests
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "storage/chroma"
COLLECTION = "varun_kb"
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

def retrieve_context(query: str) -> list[str]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(COLLECTION)
        result = collection.query(query_texts=[query], n_results=TOP_K)
        return (result.get("documents") or [[]])[0]
    except Exception:
        return []

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
        
        print(f"DEBUG: OpenAI status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"].strip()
                print(f"DEBUG: OpenAI returned: {content[:100]}...")
                return content
        
        print(f"DEBUG: OpenAI failed - Status: {response.status_code}, Response: {response.text[:200]}")
        return ""
        
    except Exception as e:
        return ""

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

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
        
        if not ctx and not os.path.exists(CHROMA_DIR):
            return jsonify({
                "error": "Knowledge base not initialized.",
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
    
    # Quick diagnostics
    print(f"API Key loaded: {bool(OPENAI_API_KEY)}")
    print(f"ChromaDB exists: {os.path.exists(CHROMA_DIR)}")
    print(f"Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
