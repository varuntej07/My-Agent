from flask import Flask, request, jsonify
import chromadb
import ollama

CHROMA_DIR = "storage/chroma"
COLLECTION = "varun_kb"
LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"

TOP_K = 4
MAX_OUT = 400

app = Flask(__name__)

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks or [])
    return f"""You are Varun's personal agent. Style: direct, confident, witty when appropriate.

Rules:
- Prefer facts from the provided context.
- If info is missing or weak, clearly signal it's a playful guess.
- Never reveal private/PII.
- No citations in the answer.
- Keep it crisp (<= 6 sentences).

User question: {query}

Context (snippets about Varun):
{context}
"""

def retrieve_context(q: str) -> list[str]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    coll = client.get_collection(COLLECTION)
    res = coll.query(query_texts=[q], n_results=TOP_K)
    return (res.get("documents") or [[]])[0]

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400
    try:
        ctx = retrieve_context(query)
        prompt = build_prompt(query, ctx)

        # stream=False so Streamlit gets a single JSON response
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.7, "num_predict": MAX_OUT},
        )
        content = resp.get("message", {}).get("content", "").strip()
        return jsonify({"answer": content, "context_used": ctx})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
