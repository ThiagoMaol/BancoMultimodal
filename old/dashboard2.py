from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from PIL import Image
import base64
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from tinydb import TinyDB
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Função para carregar imagem como base64
def pil_image_to_base64(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Banco de dados e embeddings, semelhante ao seu código anterior
# Configuração de DB e embeddings, como no código inicial (TinyDB, FAISS e OllamaEmbeddings)
db = TinyDB("image_db.json")
img_table = db.table("img_descriptions")
img_data = img_table.all()

# documents = [Document(page_content=item['content'], metadata={"source": item['source']}) for item in img_data]

documents = []
for item in img_data:
    for key, value in item.items():
        documents.append(Document(page_content=value, 
                                  metadata={"source": key}))

def cached_embedding(local=True):
    if local:
        ollama_server_url = "http://192.168.1.5:11434"
        embeddings = OllamaEmbeddings(base_url=ollama_server_url, 
                                  model='nomic-embed-text')
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return FAISS.from_documents(documents, embeddings)

print("here")
vectorstore = cached_embedding(local=False)
print("here2")

def retrieve_info(query, k=10):
    similar_response = vectorstore.similarity_search_with_score(query, k=k)
    return [s for s in similar_response if s[1] < 500]

# Layout da aplicação
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Recall Clone", className="text-center mt-4 mb-4"))),
    dbc.Row(dbc.Col(dcc.Input(id='query', type='text', placeholder='O que você procura?', style={'width': '100%'}))),
    dbc.Row(dbc.Col(dcc.Slider(id='item-slider', min=0, max=0, step=1, value=0, marks={0: '0'}, tooltip={"placement": "bottom", "always_visible": True}))),
    dbc.Row(id="gallery", className="mt-3"),
    dbc.Row(dbc.Col(html.Div(id="image-detail", className="mt-4 text-center")))
], fluid=True)

@app.callback(
    Output('gallery', 'children'),
    Output('item-slider', 'max'),
    Output('item-slider', 'marks'),
    Input('query', 'value')
)
def update_gallery(query):
    if query:
        similar_response = retrieve_info(query)
        images = [
            html.Div([
                html.Img(src=f'data:image/jpeg;base64,{pil_image_to_base64(res[0].metadata["source"])}', style={"height": "150px", "margin": "5px", "cursor": "pointer"}),
            ], id=f'img-{i}', style={'display': 'inline-block'})
            for i, res in enumerate(similar_response)
        ]
        marks = {i: str(i) for i in range(len(similar_response))}
        return images, len(similar_response) - 1, marks
    return [], 0, {0: '0'}

@app.callback(
    Output("image-detail", "children"),
    Input('item-slider', 'value'),
    Input('query', 'value')
)
def update_image_detail(selected_index, query):
    if query:
        similar_response = retrieve_info(query)
        if selected_index < len(similar_response):
            selected_image = similar_response[selected_index][0]
            image_path = selected_image.metadata["source"]
            image_data = pil_image_to_base64(image_path)
            return html.Div([
                html.Img(src=f'data:image/jpeg;base64,{image_data}', style={"width": "300px"}),
                html.H4(selected_image.metadata["source"].split("/")[-1]),
                html.P(selected_image.page_content),
            ])
    return ""

if __name__ == "__main__":
    app.run_server(debug=True)