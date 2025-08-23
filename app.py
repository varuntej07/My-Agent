import streamlit as st
import requests

st.set_page_config(page_title="Varun' Agent")
st.title("Ask Varun's assistant")

api_base = "http://127.0.0.1:8080"
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
                    if ctx:
                        for i, ch in enumerate(ctx):
                            st.markdown(f"Chunk {i+1}:\n\n{ch}")
                    else:
                        st.write("No context chunks retrieved")
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Cannot reach the API server. Make sure it's running on port 8080.")
        except requests.exceptions.Timeout:
            st.error("Timeout: Request took too long. Server might be processing.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
