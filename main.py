import os
from dotenv import load_dotenv
from rag_project.loader import load_documents_from_folder
from rag_project.chunker import chunk_and_preprocess_docs
from rag_project.embedder import embed_and_store_documents
from rag_project.retriever import load_retriever
from rag_project.prediction import create_conversational_chain
from rag_project.prediction import get_llm
from rag_project.logger import logger

# Load environment variables
load_dotenv()

# Get API Key
api_key = os.getenv("GROQ_API")

# Ensure the key is available
if not api_key:
    logger.error("GROQ_API key not found in .env file.")
    exit()

# Setup the LLM
llm = get_llm(api_key)  # Pass API Key to the function

def setup_retriever():
    try:
        # Load and preprocess documents
        docs = load_documents_from_folder()
        chunked_docs = chunk_and_preprocess_docs(docs)
        embedding_model = embed_and_store_documents(chunked_docs)
        retriever = load_retriever(embedding_model)
        qa_chain, memory = create_conversational_chain(llm, retriever)
        return qa_chain, memory
    except Exception as e:
        logger.error(f"Error in setup_retriever: {e}")

# Call the setup_retriever() function to initialize the chain
qa_chain, memory = setup_retriever()

def main():
    try:
        print("Grok ChatBot: Ask me anything (type 'exit' to quit)")
        while True:
            query = input("\n🧠 You: ")
            if query.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break

            # Get the answer from the chain
            result = qa_chain.invoke({"question": query})
            answer = result['answer']

            print(f"\n🗨️ Answer: {answer}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
