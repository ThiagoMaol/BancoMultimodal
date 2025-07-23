# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
import json
from pathlib import Path
from tinydb import TinyDB
from langchain.docstore.document import Document
from PIL import Image
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("Recall Clone")

db = TinyDB("image_db.json")
recall_table = db.table("img_descriptions")
recall_data = recall_table.all()

documents = []
for item in recall_data:
    for key, value in item.items():
        documents.append(Document(page_content=value, 
                                  metadata={"source": key}))


@st.cache_data
def cached_embedding(local=True):
    if local:
        ollama_server_url = "http://192.168.1.5:11434"
        embeddings = OllamaEmbeddings(base_url=ollama_server_url, 
                                  model='nomic-embed-text')
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return FAISS.from_documents(documents, embeddings)

vectorstore = cached_embedding()

def retrieve_info(query):
    similar_response = vectorstore.similarity_search(query, k=3)
    return [doc for doc in similar_response]


k = 10
query = st.text_input("O que você procura?")

if query:
    similar_response = vectorstore.similarity_search_with_score(query, k=k)
    similar_response = [s for s in similar_response if s[1] < 500]
    s = st.slider("item", 0, len(similar_response)-1, 0)

    st.image(Image.open(similar_response[s][0].metadata["source"]))
    st.write(similar_response[s][0].metadata["source"].split("/")[-1])
    st.write(similar_response[s][0].page_content)
