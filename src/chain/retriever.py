from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def build_retriever(store: VectorStore, k: int = 3) -> BaseRetriever:
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
