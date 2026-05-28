import structlog
from finassistant.core import setup_logging, Settings, CloudModelConfig
from ingest import load_pdf
from chain import (
    chunk_by_token,
    chunk_by_char,
    build_local_embeddings,
    build_store,
    load_store,
    build_retriever,
    build_qa_chain,
    ask,
    build_cloud_embeddings,
    build_cloud_llm,
    build_local_llm,
    store_exists,
)

setup_logging()
logger = structlog.get_logger()


def main():
    settings = Settings()
    setup_logging(settings.log_level)
    docs = load_pdf(settings.source_file)

    # pick based on model backend
    if isinstance(settings.model, CloudModelConfig):
        embeddings = build_cloud_embeddings(
            settings.model.model_name, settings.model.openai_api_key
        )
        llm = build_cloud_llm(settings.model.model_name, settings.model.openai_api_key)
        chunks = chunk_by_token(docs)
    else:
        embeddings = build_local_embeddings(settings.model.model_name)
        llm = build_local_llm(settings.model.model_name, settings.model.base_url)
        chunks = chunk_by_char(docs)

    # store init
    if store_exists(settings.db_path):
        store = load_store(
            embeddings, collection_name="statements", path=settings.db_path
        )
    else:
        chunks = chunk_by_char(docs)
        store = build_store(
            chunks, embeddings, collection_name="statements", path=settings.db_path
        )

    # retrieval + QA
    retriever = build_retriever(store)
    chain = build_qa_chain(retriever, llm)
    answer = ask(chain, "What did I spend on groceries?")
    print(f"{settings.model} model answer: {answer}")


if __name__ == "__main__":
    main()
