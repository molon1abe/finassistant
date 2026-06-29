from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def build_cloud_llm(model: str, api_key: str | None = None) -> BaseChatModel:
    return ChatOpenAI(model=model, api_key=api_key)


def build_local_llm(
    model: str,
    base_url: str = "http://localhost:11434",
    num_ctx: int = 2048,
    num_predict: int = 512,
) -> BaseChatModel:
    return ChatOllama(
        model=model, base_url=base_url, num_ctx=num_ctx, num_predict=num_predict
    )
