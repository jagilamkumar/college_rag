import os
import streamlit as st
import pandas as pd
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# Optional .env usage (if you want to keep it clean)
# from dotenv import load_dotenv
# load_dotenv()

# 🗝️ Set API key
cohere_api_key = "e8YGxF4rmdD14ghKD1UZATDY7a98zKcRO6x3JmfC"
os.environ["COHERE_API_KEY"] = cohere_api_key

# 📊 Load Excel file
excel_path = "college.xlsx"  # Put the correct filename here
df = pd.read_excel(excel_path)

# 🧾 Convert each row into LangChain Document format
documents = []
for i, row in df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=content))

# ✂️ Split content into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# 🔍 Create embeddings and vectorstore
embeddings = CohereEmbeddings(model="embed-english-v2.0")
vectorstore = FAISS.from_documents(docs, embeddings)

# 🧠 Setup LLM and QA chain
llm = ChatCohere(model="command-r")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 🖥️ Streamlit UI
st.title("📘 College Excel Info Chatbot")
query = st.text_input("Ask something about the college (e.g., departments, heads, phone numbers):")

if query:
    with st.spinner("Looking into Excel data..."):
        result = qa_chain(query)

    st.markdown("### 💬 Answer:")
    st.write(result["result"])

    with st.expander("📄 Source Context"):
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"**Row {i}:**")
            st.write(doc.page_content)
