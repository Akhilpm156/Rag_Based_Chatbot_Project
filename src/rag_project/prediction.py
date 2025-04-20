
import os
from langchain_groq import ChatGroq
from rag_project.logger import logger
from rag_project.exceptions import GenerationError
from rag_project.utils import load_config
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
#import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain_huggingface import HuggingFacePipeline


# For Groq API
##################################################

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
######################################################


# For Local LLM
########################################################

#def get_llm():
#    try:
#        config = load_config()
#
#        model_id =config['llm']['model']
#
#        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN) # use hugging face token for tokenizer model (downloading model)
#        model = AutoModelForCausalLM.from_pretrained(
#        model_id,
#        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#        device_map="auto",
#        #use_auth_token = HF_TOKEN)     # hugging face token for model (for downloading model)
#        

#        pipe = pipeline(
#        "text-generation",
#        model=model,
#        tokenizer=tokenizer,
#        max_new_tokens=512,
#        temperature=0.7,
#        top_p=0.9,
#        repetition_penalty=1.1,
#        pad_token_id=tokenizer.eos_token_id
#        )

#        llm = HuggingFacePipeline(pipeline=pipe)

#        return llm
    
#    except Exception as e:
#        logger.error(f"Failed to initialize LLM: {e}")
#        raise

############################################################

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
