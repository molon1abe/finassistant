from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader
from langchain_core.documents import Document


def load_csv(file_path: str) -> list[Document]:
    return CSVLoader(file_path).load()


def load_pdf(file_path: str) -> list[Document]:
    return PyMuPDFLoader(file_path).load()
