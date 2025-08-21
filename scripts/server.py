import os
from flask import Flask, request, jsonify
import chromadb
from flask_cors import CORS

CHROMA_DIR = "storage/chroma"
COLLECTION = "varun_kb"
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"
TOP_K = 4
MAX_OUT = 400

app = Flask(__name__)
CORS(app, origins=[
    "https://api.varuntej.dev",
    "https://varuntej.dev", 
    "http://localhost:3000"
])

def build_prompt(query: str, context_chunks: list[str]) -> str:
    # Defensive fallback for empty/weak context
    if not context_chunks or not any(chunk.strip() for chunk in context_chunks):
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
    except Exception as e:
        print(f"ChromaDB error while retrieving context: {e}", flush=True)
        return []

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "Varun's AI agent is running", "endpoints": ["/ask", "/health"]})

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()

        if not query:
            return jsonify({"error": "Missing 'query'"}), 400
        
        ctx = retrieve_context(query)
        
        # Check if ChromaDB collection is missing
        if not ctx and not os.path.exists(CHROMA_DIR):
            return jsonify({
                "error": "Knowledge base not initialized. Run 'python scripts/ingest.py' to process documents first.",
                "context_used": []
            }), 503
        
        prompt = build_prompt(query, ctx)

        # Try Ollama, fallback to defensive response if not available
        try:
            import ollama
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0.7, "num_predict": MAX_OUT},
            )
            content = resp.get("message", {}).get("content", "").strip()
            
            if not content:
                content = "Varun is still improving the model by injecting more data. This is in active development, so it may not know everything yet."
                
        except Exception as ollama_error:
            print(f"Ollama error: {ollama_error}", flush=True)
            content = "Varun is still improving the model by injecting more data. This is in active development, so it may not know everything yet."
        
        return jsonify({"answer": content, "context_used": ctx})
        
    except Exception as e:
        print(f"Server error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Varun is still improving the model by injecting more data. This is in active development, so it may not know everything yet.",
            "context_used": []
        }), 503

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
