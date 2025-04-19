
import os
from langchain_groq import ChatGroq
from rag_project.logger import logger
from rag_project.exceptions import GenerationError
from rag_project.utils import load_config
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def get_llm(api_key):
    try:
        config = load_config()
        # Initialize the LLM with the Grok API key
        llm = ChatGroq(
            model=config['llm']['model'],
            temperature=0.8,
            groq_api_key=api_key
        )
        
        return llm
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

def create_conversational_chain(llm, retriever):

    # Setup conversation memory to store the chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the ConversationalRetrievalChain using the retriever and LLM
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,  # We want to return sources for the responses
    )

    return qa_chain, memory
