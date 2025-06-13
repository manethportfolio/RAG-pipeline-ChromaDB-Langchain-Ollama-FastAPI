# ui_streamlit.py
import streamlit as st
import requests

st.title("RAG assistant for AI applications in Healthcare")

question = st.text_input("Enter your question here:")
use_rag = st.checkbox("Use RAG retrieval", value=True)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question!")
    else:
        try:
            res = requests.post(
                "http://localhost:8000/generate-response/",
                json={"question": question, "use_rag": use_rag},
                timeout=300
            )
            if res.status_code == 200:
                st.write(res.json()["response"])
            else:
                st.error(f"API error: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
