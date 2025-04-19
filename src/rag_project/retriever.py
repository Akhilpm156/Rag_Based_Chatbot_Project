from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_project.exceptions import RetrievalError
from rag_project.logger import logger
from rag_project.utils import load_config

def load_retriever(embedding_model,):
    try:
        config = load_config()
        vector_store_path = config['embedding']['file_path']
        
        vectorstore = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type=config["retriever"]["search_type"],
            search_kwargs={"k": config["retriever"]["top_k"]}
        )

        logger.info("Retriever loaded successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Error loading retriever: {e}")
        raise RetrievalError("Failed to load retriever.")
