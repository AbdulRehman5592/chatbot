from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import SecretStr
from config import GOOGLE_API_KEY
import os

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_vector_store(chunks, session_id, metadatas=None):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=SecretStr(GOOGLE_API_KEY))
    if metadatas is not None:
        vector_store = FAISS.from_texts(chunks, embedding=embedding_model, metadatas=metadatas)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embedding_model)
    vector_store.save_local(f"faiss_index/{session_id}")

def load_vector_store(session_id):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=SecretStr(GOOGLE_API_KEY))
    path = f"faiss_index/{session_id}"
    if not os.path.exists(path):
        raise Exception(f"No vector store found for session: {session_id}")
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True) 