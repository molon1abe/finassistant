from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


def build_cloud_embeddings(model: str, api_key: str | None = None) -> Embeddings:
    return OpenAIEmbeddings(model=model, api_key=api_key)


def build_local_embeddings(model: str) -> Embeddings:
    return OllamaEmbeddings(model=model)
