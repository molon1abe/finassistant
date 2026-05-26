from finassistant.core.logging import setup_logging
from ingest import PdfProcessor

setup_logging()


def main():
    file_to_read = "./rus_test.pdf"
    processor = PdfProcessor()
    pdf_data = processor.read_pdf(file_to_read)
    print(pdf_data)


if __name__ == "__main__":
    main()
