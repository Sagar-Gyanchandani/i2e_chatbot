import streamlit as st
from llama_index.core import Settings
import logging
import sys
import os.path
 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core.service_context import set_global_service_context
 
# from llama_index.llms.llama_cpp import LlamaCPP
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
# from langchain.llms import HuggingFaceHub
from llama_index.core.prompts.chat_prompts import ChatPromptTemplate, ChatMessage,MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine, ContextChatEngine, SimpleChatEngine
from llama_index.core.settings import llm_from_settings_or_context
# from llama_index.legacy.prompts import ChatPromptTemplate
# from llama_index.core.base.llms.types import ChatMessage, MessageRole
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, StorageContext, load_index_from_storage
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader, StorageContext, load_index_from_storage
 
from llama_index.llms.huggingface.base import HuggingFaceInferenceAPI

from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)

from huggingface_hub import login
 
login("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")
 
from transformers import AutoTokenizer
from llama_index.core import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from llama_index.core.memory import ChatSummaryMemoryBuffer



st.header("Chat with the I2E Bot 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."))
        llm = HuggingFaceInferenceAPI(
                generate_kwargs={"temperature": 0.0},
                model_name="meta-llama/Llama-2-70b-chat-hf"
            )
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embed_model = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
        )
        service_context=ServiceContext.from_defaults(
                    chunk_size=1000,
                    chunk_overlap=100,
                    embed_model=embed_model,
                    llm=llm
        )
        set_global_service_context(service_context)

        Settings.llm = llm
        Settings.embed_model = embed_model

        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()


from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=4098)

# chat_engine = index.as_chat_engine(chat_mode="condense_question", memory=memory, llm=Settings.llm, verbose=True)
chat_engine = index.as_chat_engine(chat_mode="condense_question", memory=memory, llm=Settings.llm, verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("++++++\n", st.session_state.messages)
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
