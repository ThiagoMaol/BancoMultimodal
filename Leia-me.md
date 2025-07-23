
# Banco de dados Multimodal
Este é um projeto que utiliza um banco de dados Multimodal para categorizar imagens automaticamente e localizá-las futuramente utilizando um algoritmo de busca semântica.

É extremamente útil para projetos onde você precisa organizar dezenas de imagens ou documentos por tags e conseguir buscar no futuro de maneira eficiente.

Este é o código fonte do projeto apresentado neste vídeo:
[https://www.instagram.com/reel/C8Ndmh2OAze/](https://www.instagram.com/reel/C_qDoe2O1Dn/)

Este conteúdo é abordado em nossa [Trilha Aplicações IA com Python](https://asimov.academy/trilha-aplicacoes-ia-com-python/). Acesse o link e conheça mais.


# Requisitos
- `Python 3.6+`

# Instalação
1.	Clone o repositório e navegue até o diretório do projeto.
2.	Instale os pacotes Python necessários:

`pip install -r requirements.txt`

3.	Crie um arquivo .env e adicione suas chaves GOOGLE_API_KEY e OPENAI_API_KEY.
4.  Adicione suas imagens na pasta images e rode o arquivo `1. milvus_mmdb.py`. Ele irá comprimi-las 
5. O script `2. dashboard.py` é o dashboard em si onde você pode realizar buscas por texto ou por outras imagens.

