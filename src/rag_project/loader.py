from pathlib import Path
from rag_project.logger import logger
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from rag_project.exceptions import DocumentLoadingError
from rag_project.utils import load_config
import warnings
warnings.filterwarnings('ignore')

def load_documents_from_folder():

    config = load_config()
    folder_path = config['paths']['raw_data']
    docs = []
    folder = Path(folder_path)
    try:
        for file in folder.iterdir():
            if file.suffix == ".pdf":
                loader = PyPDFLoader(file_path=str(file))
            elif file.suffix == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path=str(file))
            elif file.suffix == ".csv":
                loader = CSVLoader(file_path=str(file))
            elif file.suffix == ".xlsx":
                loader = UnstructuredExcelLoader(file_path=str(file))
            else:
                logger.warning(f"Unsupported file format: {file.name}")

                continue

            loaded_docs = loader.load()
            logger.info(f"Loaded {len(loaded_docs)} docs from {file.name}")
            
            docs.extend(loaded_docs)
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise DocumentLoadingError("Failed to load documents.") from e

    if not docs:
        raise DocumentLoadingError("No documents loaded. Check the folder path and file types.")

    return docs
