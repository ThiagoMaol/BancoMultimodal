
import streamlit as st
from dotenv import load_dotenv
import os
from pymilvus import MilvusClient
from tempfile import NamedTemporaryFile
from encoder import Encoder


load_dotenv()
st.set_page_config(layout="wide")


@st.cache_resource
def get_encoder():
    model_name = "BAAI/bge-base-en-v1.5"
    model_path = "./Visualized_base_en_v1.5.pth"  # Change to your own value if using a different model path
    return Encoder(model_name, model_path)


@st.cache_data
def query_db(query_text, image_path=None):
    query_image = os.path.join(
        "compressed_images", "pexels-christian-wallner-5611088.jpg"
    )  

    if image_path and len(query_text) > 0:
        query_vec = encoder.encode_query(image_path=image_path, text=query_text)

    elif len(query_text) == 0:
        query_vec = encoder.encode_image(image_path=image_path)

    else:
        query_vec = encoder.encode_text(text=query_text)

    collection_name = "multimodal_rag_demo"
    milvus_client = MilvusClient(uri="databases/milvus_demo.db")


    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vec],
        output_fields=["image_path"],
        limit=50,  # Max number of search results to return
        search_params={"metric_type": "COSINE", "params": {}},  # Search parameters
    )[0]

    retrieved_images = [hit.get("entity").get("image_path") for hit in search_results]
    return retrieved_images

encoder = get_encoder()
query_image = st.sidebar.file_uploader("Escolha uma imagem", type=["jpg", "png", "jpeg"])
query_text = st.sidebar.text_input("O que você procura?")

image_path = None

if query_image:
    with NamedTemporaryFile(dir='.', suffix='.jpg') as f:
        f.write(query_image.getbuffer())
        image_path = f.name
        retrieved_images = query_db(query_text, image_path)
elif query_text:
    retrieved_images = query_db(query_text, image_path)



n = 3
if query_text or query_image:
    num_rows = int(len(retrieved_images) / n)

    for i in range(num_rows):
        cols = st.columns(n)
        for j in range(n):
            with cols[j]:
                st.image(retrieved_images[i * n + j])