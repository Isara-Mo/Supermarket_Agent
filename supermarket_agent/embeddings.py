import streamlit as st
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
from .config import dashscope_api_key


@st.cache_resource
def init_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=dashscope_api_key
    )


@st.cache_resource
def init_llm():
    return init_chat_model("deepseek-chat", model_provider="deepseek")
