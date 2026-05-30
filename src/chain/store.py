import os
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "db"


def get_client(db_path: str) -> ClientAPI:
    return chromadb.PersistentClient(path=db_path)


def build_store(
    docs: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    path: str = str(DEFAULT_DB_PATH),
) -> VectorStore:
    if not docs:
        raise ValueError("Cannot build store from empty document list")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=path,
    )
    vector_store.add_documents(documents=docs)
    return vector_store


def load_store(
    embeddings: Embeddings,
    collection_name: str,
    path: str = str(DEFAULT_DB_PATH),
) -> VectorStore:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=path,
    )


def store_exists(store_path: str) -> bool:
    return os.path.exists(store_path)


def doc_exists(store: VectorStore, file_id: str) -> bool:
    results = store.get(where={"file_id": file_id})
    return len(results["ids"]) > 0


def add_documents(store: VectorStore, docs: list[Document]) -> None:
    if not docs:
        raise ValueError("Cannot add empty document list to store")
    store.add_documents(documents=docs)
