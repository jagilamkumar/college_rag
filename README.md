# 🎓 College Info Chatbot

A conversational AI chatbot built using [LangChain](https://www.langchain.com/), [Cohere](https://cohere.com/), and [Streamlit](https://streamlit.io/) that answers questions based on college-related information extracted from a PDF document.

## 🧠 Features

- 💬 Ask questions about the college (courses, fees, facilities, contacts, etc.)
- 📄 Uses content from a PDF file (`college_info.pdf`) as its knowledge base
- 🔍 Intelligent document retrieval with FAISS vector store
- 🤖 Powered by Cohere's `command-r` LLM for natural answers
- 🎙️ Optional voice input support (if enabled)
- 🖥️ Beautiful and interactive UI with Streamlit
- ☁️ Ready for deployment on Render

## 🚀 Live Demo

_You can host this app on [Render](https://render.com), [Streamlit Cloud](https://streamlit.io/cloud), or any cloud provider supporting Python apps._

## 📁 Project Structure

```
college-chatbot/
│
├── college_info.pdf             # Your knowledge base (customizable)
├── college_bot.py               # Main Streamlit application
├── requirements.txt             # Python dependencies
├── .env (optional)              # Store API keys securely (not committed)
└── README.md                    # Project documentation
```

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jagilamkumar/college_rag.git
cd college_bot
```

### 2. Install Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Your Cohere API Key

Edit the `.env` file or directly add in `college_bot.py`:

```python
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"
```

Or use `.env`:

```env
COHERE_API_KEY=your-cohere-api-key
```

### 4. Run the App

```bash
streamlit run college_bot.py
```

## ✨ Optional Features

- 🔊 Voice input (via browser microphone)
- 📊 Upload XLSX instead of PDF (customizable with Pandas)
- 🧠 Other LLMs or vector stores (Chroma, Qdrant, etc.)

## 🌍 Deployment on Render

1. Push code to GitHub
2. Create a new **Web Service** on [render.com](https://render.com/)
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run college_bot.py --server.port 10000`
   - **Environment Variable:** `COHERE_API_KEY=your-key`
4. Deploy and visit your hosted chatbot!

## ✅ Requirements

- Python 3.9+
- A valid Cohere API key
- Streamlit
- LangChain + LangChain-Community
- FAISS for vector storage

## 🙌 Credits

- [Cohere](https://cohere.com/) – for LLM and embedding services
- [LangChain](https://www.langchain.com/) – for chaining and document processing
- [Streamlit](https://streamlit.io/) – for interactive frontend
- [FAISS](https://github.com/facebookresearch/faiss) – for fast vector similarity search

## 📜 License

MIT License – Free to use, modify, and distribute.
