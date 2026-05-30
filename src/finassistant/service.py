import structlog
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from finassistant.core.config import Settings, CloudModelConfig
from ingest import load_pdf, load_csv, file_uuid
from chain import (
    chunk_by_token,
    chunk_by_char,
    build_local_embeddings,
    load_store,
    build_retriever,
    build_qa_chain,
    build_cloud_embeddings,
    build_cloud_llm,
    build_local_llm,
    doc_exists,
    add_documents,
)

logger = structlog.get_logger()


def init_store(settings: Settings, embeddings: Embeddings) -> VectorStore:
    store = load_store(embeddings, collection_name="statements", path=settings.db_path)
    logger.info("Store ready", path=settings.db_path)
    return store


def add_document(settings: Settings, store: VectorStore, file_path: str):
    if doc_exists(store, file_uuid(file_path)):
        logger.info("Document already ingested, skipping", path=file_path)
        return
    docs = load_pdf(file_path) if file_path.endswith(".pdf") else load_csv(file_path)
    chunks = (
        chunk_by_token(docs)
        if isinstance(settings.model, CloudModelConfig)
        else chunk_by_char(docs)
    )
    add_documents(store, chunks)


def init_model(settings: Settings) -> BaseChatModel:
    if isinstance(settings.model, CloudModelConfig):
        llm = build_cloud_llm(settings.model.model_name, settings.model.openai_api_key)
    else:
        llm = build_local_llm(settings.model.model_name, settings.model.base_url)
    return llm


def build_embeddings(settings: Settings) -> Embeddings:
    if isinstance(settings.model, CloudModelConfig):
        embeddings = build_cloud_embeddings(
            settings.model.model_name, settings.model.openai_api_key
        )
    else:
        embeddings = build_local_embeddings(settings.model.embed_model)
    return embeddings


def build_chain(store: VectorStore, llm: BaseChatModel) -> Runnable:
    retriever = build_retriever(store)
    return build_qa_chain(retriever, llm)
