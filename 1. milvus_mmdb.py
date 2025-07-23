import torch
from FlagEmbedding.visual_bge.modeling import Visualized_BGE
# from visual_bge.modeling import Visualized_BGE
import os
from tqdm import tqdm
from glob import glob
from encoder import Encoder
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
Image.MAX_IMAGE_PIXELS = 500000000
from pymilvus import MilvusClient

test = False
model_name = "BAAI/bge-base-en-v1.5"
model_path = "./Visualized_base_en_v1.5.pth"
encoder = Encoder(model_name, model_path)



def resize_image(input_path, output_path, max_width):
    with Image.open(input_path) as img:
        width, height = img.size
        if width > max_width:
            new_width = max_width
            new_height = int((new_width / width) * height)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.save(output_path, optimize=True, quality=85)

def process_image(input_path, output_dir):
    max_width=600
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    resize_image(input_path, output_path, max_width)
    # print(f"Imagem comprimida: {filename}")


def compress_images(input_dir, output_dir):    
    caminhos = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            caminhos.append(os.path.join(root, file))
    image_files = [f for f in caminhos if f.split("/")[-1] not in os.listdir(output_dir)]
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda image: process_image(image, output_dir), image_files),
                            total=len(image_files),
                            desc="Comprimindo imagens",
                            unit="imagem",
                            ncols=100))

input_dir = "imagens"
output_dir = "compressed_images"
compress_images(input_dir, output_dir)




data_dir = ("compressed_images")
image_list = glob(
    os.path.join(data_dir, "*.jpg")
) 
image_dict = {}
for image_path in tqdm(image_list, desc="Generating image embeddings: "):
    try:
        image_dict[image_path] = encoder.encode_image(image_path)
    except Exception as e:
        print(f"Failed to generate embedding for {image_path}. Skipped.")
        continue
print("Number of encoded images:", len(image_dict))




dim = len(list(image_dict.values())[0])
collection_name = "multimodal_rag_demo"
milvus_client = MilvusClient(uri="databases/milvus_demo.db")

milvus_client.create_collection(
    collection_name=collection_name,
    auto_id=True,
    dimension=dim,
    enable_dynamic_field=True,
)

milvus_client.insert(
    collection_name=collection_name,
    data=[{"image_path": k, "vector": v} for k, v in image_dict.items()],
)


# Teste de busca
if test == True:
    query_image = os.path.join(
        data_dir, "pexels-christian-wallner-5611088.jpg"
    ) 
    query_text = "A toy of this"
    # query_text = "Spash and Fruit"

    query_vec = encoder.encode_query(image_path=query_image, text=query_text)
    # query_vec = encoder.encode_text(text=query_text)

    search_results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vec],
        output_fields=["image_path"],
        limit=9,  # Max number of search results to return
        search_params={"metric_type": "COSINE", "params": {}},  # Search parameters
    )[0]

    retrieved_images = [hit.get("entity").get("image_path") for hit in search_results]
    Image.open(retrieved_images[4])