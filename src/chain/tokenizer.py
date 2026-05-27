from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_faiss import FAISS


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


def vectorise_docs(
    docs: list[Document], model: str = "text-embedding-3-large"
) -> FAISS:
    if not docs:
        raise ValueError("Cannot vectorise empty document list")

    embeddings = OpenAIEmbeddings(model=model)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore
