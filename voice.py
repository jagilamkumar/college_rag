import os
import streamlit as st
import speech_recognition as sr

from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# --- 🔑 COHERE API KEY ---
os.environ["COHERE_API_KEY"] = "e8YGxF4rmdD14ghKD1UZATDY7a98zKcRO6x3JmfC"

# --- 📄 Load PDF File ---
pdf_path = "college_info.pdf"  # Make sure this is in your working directory
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# --- ✂️ Split Text into Chunks ---
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# --- 🔍 Create Vector Store ---
embeddings = CohereEmbeddings(model="embed-english-v2.0")
vectorstore = FAISS.from_documents(docs, embeddings)

# --- 🤖 Setup LLM RetrievalQA ---
llm = ChatCohere(model="command-r")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# --- 🎙️ Voice Input Function ---
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("🎤 Listening... please speak clearly.")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"📝 You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError:
        st.error("API unavailable or quota exceeded.")

    return ""

# --- 🖥️ Streamlit UI ---
st.set_page_config(page_title="College Chatbot", page_icon="🎓")
st.markdown("<h1 style='text-align: center; color: navy;'>🎓 College Info Chatbot</h1>", unsafe_allow_html=True)

# 📥 Text Input or Voice Input
query = st.text_input("💬 Type your question about the college:")
use_mic = st.button("🎙️ Use Voice Instead")

if use_mic:
    query = recognize_speech_from_mic()

if query:
    with st.spinner("🔍 Searching..."):
        result = qa_chain(query)

    st.markdown("### ✅ Answer:")
    st.write(result["result"])

    with st.expander("📄 Source Context"):
        for doc in result["source_documents"]:
            st.markdown(doc.page_content)

# --- Optional: Add a footer ---
st.markdown("""
<hr>
<div style='text-align: center; color: grey;'>
    Made with 💙 using Cohere + LangChain + Streamlit
</div>
""", unsafe_allow_html=True)
