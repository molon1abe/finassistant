from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    return text_splitter.split_documents(docs)


def chunk_by_char(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    if not docs:
        raise ValueError("Cannot chunk empty document list")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
