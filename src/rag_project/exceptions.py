class DocumentLoadingError(Exception):
    """Custom exception for document loading errors."""
    pass

class EmbeddingError(Exception):
    """Custom exception for embedding errors."""
    pass

class RetrievalError(Exception):
    """Custom exception for retrieval failures."""
    pass

class GenerationError(Exception):
    """Custom exception for generation step failures."""
    pass
