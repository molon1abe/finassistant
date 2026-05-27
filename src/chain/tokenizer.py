from pathlib import Path
from typing import Any, Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "db"


class VectorStoreClass(Protocol):
    @classmethod
    def from_documents(
        cls, documents: list[Document], embedding: Any, **kwargs: Any
    ) -> VectorStore: ...


def chunk_by_token(
    docs: list[Document],
    encoding: str = "cl100k_base",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    if not docs:
        raise ValueError("Cannot chunk empty document list")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs


def _vectorise_docs(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    vectorstore_cls: type[VectorStoreClass] = Chroma,
    **kwargs,
) -> VectorStore:
    """
    Docs vectorisation wrapper with embeddings
    Supports FAISS, ChromaDB
    """
    if not docs:
        raise ValueError("Cannot vectorise empty document list")
    return vectorstore_cls.from_documents(docs, embeddings, **kwargs)


def vectorise_chroma(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    persist_directory: Path = DEFAULT_DB_PATH,
) -> VectorStore:
    """Vectorise with ChromaDB"""
    return _vectorise_docs(
        docs=docs,
        embeddings=embeddings,
        vectorstore_cls=Chroma,
        persist_directory=persist_directory,
    )
