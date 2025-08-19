import streamlit as st
import requests

st.set_page_config(page_title="Varun' Agent")
st.title("Ask Varun's assistant")

api_base = "http://127.0.0.1:8000"
query = st.text_input("Ask about Varun:", value="", placeholder="e.g., What projects has Varun built recently?")

if st.button("Ask") and query.strip():
    with st.spinner("Thinking..."):
        try:
            r = requests.post(f"{api_base}/ask", json={"query": query}, timeout=120)
            if r.status_code == 200:
                data = r.json()
                st.subheader("Answer")
                st.write(data.get("answer", ""))
                with st.expander("Debug: retrieved context"):
                    ctx = data.get("context_used") or []
                    for i, ch in enumerate(ctx):
                        st.markdown(f"**Chunk {i+1}:**\n\n{ch}")
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
