import streamlit as st
import os
import chromadb
from embeddings import GeminiEmbeddingFunction

# Initialize ChromaDB client
# doc_embed_fn = GeminiEmbeddingFunction(document_mode=True)
# query_embed_fn = GeminiEmbeddingFunction(document_mode=False)
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="skillsdb", embedding_function=embed_fn)

# App title and style
st.set_page_config(page_title="💼 Skills & Roles Search", layout="wide")
st.markdown("""
    <style>
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #6f42c1;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 AI-powered Skills & Roles Finder")

# Sidebar: Add documents
with st.sidebar:
    st.header("📄 Add a Role Description")
    new_doc = st.text_area("Paste job description or role profile", height=200)
    doc_id = st.text_input("Optional ID")
    if st.button("Add Document"):
        if new_doc.strip():
            collection.add(documents=[new_doc], ids=[doc_id or str(hash(new_doc))])
            st.success("✅ Document added successfully!")
        else:
            st.warning("Please paste a role description first.")

# Main: Search for relevant documents
st.subheader("🔍 Search for Skills and Responsibilities")
query = st.text_input("What do you want to know?", placeholder="e.g., Skills needed for a Systems Administrator")

if st.button("Search"):
    if query.strip():
        embed_fn.document_mode = False
        print("Querying : ", query)
        result = collection.query(query_texts=[query], n_results=3)
        docs = result["documents"][0]
        st.markdown("### ✨ Relevant Information:")
        for i, doc in enumerate(docs):
            st.markdown(f"**Result {i+1}:**")
            st.markdown(f"> {doc}")
            st.markdown("---")
    else:
        st.warning("Please enter a query.")

st.caption("🚀 Powered by Gemini + ChromaDB + Streamlit")
