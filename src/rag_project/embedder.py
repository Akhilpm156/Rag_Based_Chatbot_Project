from langchain_huggingface import HuggingFaceEmbeddings
import tiktoken
from langchain_chroma import Chroma
import torch
from rag_project.exceptions import EmbeddingError
from rag_project.logger import logger
from rag_project.utils import load_config
import os


def embed_and_store_documents(documents):
    try:
        config = load_config()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = config['embedding']['model_name']
        vector_store_path = config['embedding']['file_path']

        logger.info(f"Embedding model: {model_name}")

        logger.info(f"Device: {device}")

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True, "device": device}
        )

        # Check if the vector store exists
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            logger.info("Vector store already exists. Loading it...")
            vectorstore = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding_model
            )
        else:
            logger.info("Vector store not found. Creating a new one...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=vector_store_path
            )
            logger.info(f"Documents embedded and stored successfully: {len(documents)}")

        return embedding_model

    except Exception as e:
        logger.exception("Failed during embedding and storing process")
        raise EmbeddingError("Embedding or storing documents failed") from e