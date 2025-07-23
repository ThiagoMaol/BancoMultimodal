from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_community.embeddings import OllamaEmbeddings


# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI = "./milvus_example.db"
ollama_server_url = "http://192.168.1.5:11434"
embeddings = OllamaEmbeddings(base_url=ollama_server_url, 
                                  model='nomic-embed-text')

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
)


vector_store_saved = Milvus.from_documents(
    [Document(page_content="foo!")],
    embeddings,
    collection_name="langchain_example",
    connection_args={"uri": URI},
)


def cached_embedding(local=True):
    if local:
        ollama_server_url = "http://192.168.1.5:11434"
        embeddings = OllamaEmbeddings(base_url=ollama_server_url, 
                                  model='nomic-embed-text')
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return FAISS.from_documents(documents, embeddings)

vectorstore = cached_embedding()