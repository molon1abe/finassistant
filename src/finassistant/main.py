import structlog
from finassistant.core.logging import setup_logging
from ingest import load_pdf, load_csv


setup_logging()
logger = structlog.get_logger()


def main():
    pdf_file_to_read = "./raycast_1.pdf"
    csv_file_to_read = "./tests/data/bank_statement.csv"
    pdf_content = load_pdf(pdf_file_to_read)
    csv_content = load_csv(csv_file_to_read)
    logger.debug(f"Content loaded from {pdf_file_to_read} file:\n{pdf_content}")
    logger.info("Loaded pdf file", pdf_file=pdf_file_to_read)
    logger.debug(f"Content loaded from {csv_file_to_read} file:\n{csv_content}")
    logger.info("Loaded csv file", csv_file=csv_file_to_read)


if __name__ == "__main__":
    main()
