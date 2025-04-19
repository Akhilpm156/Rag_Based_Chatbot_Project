import os
import gradio as gr
from rag_project.loader import load_documents_from_folder
from rag_project.chunker import chunk_and_preprocess_docs
from rag_project.embedder import embed_and_store_documents
from rag_project.retriever import load_retriever
from rag_project.prediction import create_conversational_chain
from rag_project.prediction import get_llm
from rag_project.logger import logger
from dotenv import load_dotenv

# For Groq API
#######################################
# Load environment variables
load_dotenv()

# Get API Key
api_key = os.getenv("GROQ_API")

# Ensure the key is available
if not api_key:
    logger.error("GROQ_API key not found in .env file.")
    exit()

# Setup the LLM API
llm = get_llm(api_key)  # Pass API Key to the function
########################################

# For Local LLM
########################################

#llm = get_llm()

########################################

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
        return None

# Call the setup_retriever() function to initialize the chain
qa_chain, memory= setup_retriever()

def answer_question(question):
    """Function that takes a question and returns an answer using the RAG chain."""
    try:
        if qa_chain is not None:
            result = qa_chain.invoke({"question": question})
            answer = result['answer']
            
            return answer
        else:
            return "Error: QA Chain not initialized", []
    except Exception as e:
        logger.error(f"Error in answering the question: {e}")
        return "Error: An unexpected issue occurred.", []

# Gradio Interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 Conversational AI with Grok")

        question_input = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
        answer_output = gr.Textbox(label="Answer", interactive=False)

        generate_button = gr.Button("🔍 Generate Answer", variant="primary")
        reset_button = gr.Button("🔁 Reset Chat Context", variant="secondary")

        # Answering the question
        generate_button.click(
            fn=answer_question,
            inputs=question_input,
            outputs=answer_output
        )

        # Resetting the memory
        def reset_chat_context():
            global memory
            if memory:
                memory.clear()
                return "✅ Chat context reset. You can start a new conversation."
            return "⚠️ Memory not initialized."

        reset_button.click(
            fn=reset_chat_context,
            inputs=[],
            outputs=[answer_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
