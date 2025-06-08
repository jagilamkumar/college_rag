import os
from dotenv import load_dotenv
import streamlit as st
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# Load environment variables from .env file (optional, but recommended)
load_dotenv()

# Get Cohere API key from environment variable
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    st.error("❗ Cohere API key not found. Please set COHERE_API_KEY in your environment or .env file.")
    st.stop()

os.environ["COHERE_API_KEY"] = cohere_api_key

# Load the PDF file (make sure this is in your working directory)
pdf_path = "college_info.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Split text into manageable chunks for embedding
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Initialize Cohere embeddings with required model parameter
embeddings = CohereEmbeddings(model="embed-english-v2.0")

# Create FAISS vectorstore from documents and embeddings
vectorstore = FAISS.from_documents(docs, embeddings)

# Initialize Cohere Chat model
llm = ChatCohere(model="command-r")

# Setup RetrievalQA chain to answer queries based on documents
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Streamlit UI
st.title("🎓  NNRG College Info Chatbot")
query = st.text_input("Ask something about the college (e.g., courses, fees, contact):")

if query:
    with st.spinner("Searching for answer..."):
        result = qa_chain(query)

    st.markdown("**Answer:**")
    st.write(result["result"])

    with st.expander("📄 Source Context"):
        for doc in result["source_documents"]:
            st.write(doc.page_content)
