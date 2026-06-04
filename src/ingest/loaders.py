import hashlib
import uuid
from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader
from langchain_core.documents import Document


def file_uuid(file_path: str) -> str:
    digest = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_OID, digest))


def load_csv(file_path: str) -> list[Document]:
    docs = CSVLoader(file_path).load()
    file_id = file_uuid(file_path)
    for doc in docs:
        doc.metadata["file_id"] = file_id
    return docs


def load_pdf(file_path: str) -> list[Document]:
    docs = PyMuPDFLoader(file_path).load()
    file_id = file_uuid(file_path)
    for doc in docs:
        doc.metadata["file_id"] = file_id
    return docs
