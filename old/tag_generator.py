import time
import os
from PIL import Image
from datetime import datetime
from image_processer import ImageInference
from threading import Thread
from tinydb import TinyDB
import pprint


def listar_arquivos(diretorio):
    caminhos = []
    for root, _, files in os.walk(diretorio):
        for file in files:
            caminhos.append(os.path.join(root, file))
    return caminhos


def describe_image(img_path):
    description = image_inference.process_image(img_path, prompt)
    img_table.insert({
        img_path: description.content
    })



if __name__ == "__main__":
    # image_inference = ImageInference(model="local")
    image_inference = ImageInference(model="gemini")

    try:
        db = TinyDB("image_db.json")
        img_table = db.table("img_descriptions")
        images = listar_arquivos("imagens")
        prompt = open("prompts/description_prompt.md", "r").read()

        img_paths = []
        for img in images:
            if img not in [list(i.keys())[0] for i in img_table.all()]:
                if img.split(".")[-1].lower() in ["jpg", "png", "jpeg"]:
                    img_paths.append(img)    
        
        batch_size = 10
        for i in range(0, len(img_paths), batch_size):
            print(f"{i} - {len(img_paths)/batch_size}")
            batch_imgs = img_paths[i:i+batch_size]
            responses = image_inference.batch_process(batch_imgs, 
                                                      prompt)
            
            for img, response in zip(batch_imgs, responses):
                img_table.insert({
                    img: response.content
                })
            
            # i = 0
            # pprint.pp(responses[i].content)
            # Image.open(batch_imgs[i])
            
        


    except KeyboardInterrupt:
        print("Captura de tela interrompida pelo usuário.")