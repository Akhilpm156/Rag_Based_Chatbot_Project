# 🤖 RAG Conversational Chatbot using Groq API (LLaMA 3)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that answers user questions based on preloaded documents from a local folder. It combines context-aware retrieval with language generation using **Groq's API** powered by **Meta’s LLaMA 3** model.

---

## 🚀 Features

- Conversational question-answering with memory
- Automatically loads and processes documents from a folder
- Token-aware chunking and embedding
- Embedding storage and retrieval using ChromaDB
- Answer generation via Groq’s LLaMA 3 API
- User interface built with Gradio
- Memory reset functionality for new conversation context

---

## 🧪 Methodology

1. **Document Loading**: Loads all `.docx`, `.xlsx`, `.pdf`, and supported file types from a local folder using `langchain` loaders.
2. **Chunking**: Uses `tiktoken` to split large texts into optimized, token-aware chunks.
3. **Embedding**: Each chunk is embedded using a transformer-based embedding model.
4. **Vector Store**: All embeddings are stored and indexed in **ChromaDB**.
5. **Retrieval**: When a question is asked, relevant document chunks are retrieved using similarity search.
6. **LLM Generation**: The context and user question are passed to **LLaMA 3 (via Groq API)** for response generation.

---

## 🧠 LLM Used

- **Model**: LLaMA 3 (via [Groq API](https://console.groq.com/))
- **Provider**: GroqCloud
- **Why**: High-speed hosted inference, no local model or GPU required

---

## 🔍 Embedding Model

- **Model**: `nomic-ai/nomic-embed-text-v1` from langchain HuggingFaceEmbeddings
- **Purpose**: Converts document chunks into vector embeddings
- **Storage**: Embeddings stored in a local **ChromaDB** instance

---

## 📁 Project Structure

```
├── artifacts
├── data
├── logs
├── research 			   # Jupyter Notebook with local Llama3 Model
├── src/
│   └── rag_project/
│	├── __init__.py
│	├── loader.py              # Load documents from folder
│	├── chunker.py             # Split documents into token-aware chunks
│	├── embedder.py            # Embed and store chunks
│	├── retriever.py           # Load retriever from vector DB
│	├── prediction.py          # Setup QA chain with memory
│	├── logger.py              # Logging setup
│	├── utils.py 		   # Helper functions
│	├── exceptions.py 	   # Custom exceptions
├── app.py                         # Gradio app with resettable memory
├── .env                           # API key configuration
├── config.yaml			   # Configuration file
├── setup.py			   # Python package
├── requirements.txt
├──README.md
```

---

## 💻 How to Run

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up `.env` file**:
   ```
   GROQ_API=your_groq_api_key_here
   ```

3. **Ensure your documents are in the correct folder** (e.g., `./data/`)

4. **Run the app**:
   ```bash
   python app.py
   ```

5. **Open Gradio web interface** for windows :127.0.0.1:7860 to start chatting!

---

## 🧠 Key Design Notes

- Uses Groq-hosted LLaMA 3 for real-time responses
- No local LLM is used (ideal for low-resource machines)
- Memory management allows context-aware conversation
- Reset functionality lets users start fresh chats easily

---

## 📚 References

- [Groq API](https://console.groq.com/)
- [LangChain](https://docs.langchain.com/)
- [Huggingface](https://huggingface.co/docs)
- [ChromaDB](https://www.trychroma.com/)
- [Gradio](https://www.gradio.app/docs)

---

## 🧑‍💻 Author

Developed by Akhil P M  
This project was built as part of an AI/ML evaluation task.