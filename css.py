import os
import streamlit as st
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# Setup API
os.environ["COHERE_API_KEY"] = "e8YGxF4rmdD14ghKD1UZATDY7a98zKcRO6x3JmfC"

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f5f6fa;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2.8em;
    }
    .stTextInput>div>div>input {
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 8px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load PDF and prepare Vectorstore ---
pdf_path = "college_info.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

embeddings = CohereEmbeddings(model="embed-english-v2.0")
vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatCohere(model="command-r")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# --- UI Layout ---
st.title("🎓 College Info Chatbot")

with st.container():
    query = st.text_input("🧾 Ask me anything about the college:")
    if st.button("🔍 Search"):
        if query.strip() == "":
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("🤖 Thinking..."):
                result = qa_chain(query)

            st.success("✅ Answer found!")

            st.markdown("### 💬 Answer:")
            st.write(result["result"])

            with st.expander("📚 Show Source Text"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Page {i}:**")
                    st.write(doc.page_content)
