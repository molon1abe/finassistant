import structlog
from finassistant.core import setup_logging, Settings
from chain import ask
from .service import init_store, add_document, init_model, build_embeddings, build_chain


setup_logging()
logger = structlog.get_logger()


def main():
    settings = Settings()
    setup_logging(settings.log_level)
    embeddings = build_embeddings(settings)
    store = init_store(settings, embeddings)
    add_document(settings, store, settings.source_file)
    model = init_model(settings)
    chain = build_chain(store, model)
    answer = ask(chain, "What did I spend on groceries?")
    print(f"{settings.model} model answer: {answer}")


if __name__ == "__main__":
    main()
