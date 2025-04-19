import re
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from rag_project.utils import preprocess_text
from rag_project.exceptions import EmbeddingError
from typing import List
from rag_project.utils import load_config
from rag_project.logger import logger

def chunk_and_preprocess_docs(docs: List[Document]) -> List[Document]:
    
    try:
        config = load_config()
        chunk_size = config['chunking']['chunk_size']
        chunk_overlap = config['chunking']['chunk_overlap']
        encoding = config['chunking']['encoding']

        logger.info(f"Chunking config: size={chunk_size}, overlap={chunk_overlap}, encoding={encoding}")

        token_splitter = TokenTextSplitter(
            encoding_name=encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunked_documents = []

        for doc in docs:
            cleaned_text = preprocess_text(doc.page_content)
            doc.page_content = cleaned_text

            chunks = token_splitter.split_text(cleaned_text)

            for i, chunk in enumerate(chunks, 1):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "unknown"),
                        "chunk": i
                    }
                )
                chunked_documents.append(chunked_doc)
        

        logger.info(f"Chunking finished. Total chunks: {len(chunked_documents)}")
        return chunked_documents

    except Exception as e:
        logger.exception("Error during chunking")
        raise EmbeddingError("Chunking failed") from e