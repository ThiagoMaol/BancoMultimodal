import base64
from openai import OpenAI
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import os
import cv2


load_dotenv()
Image.MAX_IMAGE_PIXELS = 200000000

class ImageInference:
    def __init__(self, 
                 model="local"):
        
        if model == "gemini":
            self.chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        else:
            base_url="http://192.168.1.5:23333/v1"
            self.client = OpenAI(api_key='YOUR_API_KEY', 
                                base_url=base_url)
            
            model_name = self.client.models.list().data[0].id
            self.chat = ChatOpenAI(
                temperature=0.0, 
                model=model_name,
                base_url=base_url, 
                api_key="YOUR_API_KEY",
            )


    @staticmethod
    def _load_image(image_path):

        # max_size = (1000, 1000)
        # with Image.open(img_paths[0]) as img:
        #     img.thumbnail(max_size, Image.LANCZOS)  # Reduz a imagem ao carregar
        #     buffered = BytesIO()
        #     img.save(buffered, format="JPEG")
        #     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        with Image.open(image_path) as pil_image:
            format = "JPEG" if image_path.split(".")[-1] in ["jpeg", "jpg"] else "PNG"
            buffered = BytesIO()

            # print(image_path)
            resize_factor = int(pil_image.width / 1000) + 1
            pil_image = pil_image.resize((pil_image.width // resize_factor, 
                                        pil_image.height // resize_factor))
            
            pil_image.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    @staticmethod
    def load_image_cv2(image_path, max_width=800):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        resize_factor = max(1, img.shape[1] // max_width)
        img_resized = cv2.resize(img, (img.shape[1] // resize_factor, img.shape[0] // resize_factor))

        _, buffer = cv2.imencode('.jpg', img_resized)
        img_str = base64.b64encode(buffer).decode("utf-8")
        return img_str

    def process_image(self, 
                    img_path,
                    prompt
                    ):
        img = self._load_image(img_path)
        input = [
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                ])
            ]
        
        return self.chat.invoke(input)

    def batch_process(self, img_paths, prompt):
        # for i, img_path in enumerate(img_paths):
        #     print(i, len(img_paths), img_path)
        #     img = self._load_image(img_path) # self.load_image_cv2(img_path)

        with ProcessPoolExecutor() as executor:
            resultados = list(executor.map(self.load_image_cv2, img_paths))
            # resultados = list(executor.map(self._load_image, img_paths))

        inputs = []
        for img_base_64 in resultados:
            inputs+= [[HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{img_base_64}"}},
            ])]]
        len(inputs)

        return self.chat.batch(inputs)


if __name__ == "__main__":
    self = ImageInference(model="local")
