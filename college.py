import os
import streamlit as st
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# 🔑 Set your Cohere API key
os.environ["COHERE_API_KEY"] = "e8YGxF4rmdD14ghKD1UZATDY7a98zKcRO6x3JmfC"  # Replace with your actual key

# 📄 Load your PDF
pdf_path = "college_info.pdf"  # Ensure the file is in the project directory
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# ✂️ Split into manageable chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 🔍 Convert text chunks into vector database
embeddings = CohereEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# 🤖 Create the QA chain with Cohere LLM
llm = ChatCohere(model="command-r")  # Other options: command-r+, command-light
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 🖥️ Streamlit UI
st.title("🎓 College Information Chatbot")
query = st.text_input("Ask a question about the college:")

if query:
    result = qa_chain(query)
    st.markdown("**Answer:**")
    st.write(result["result"])

    with st.expander("🔍 Source context"):
        for doc in result["source_documents"]:
            st.write(doc.page_content)
