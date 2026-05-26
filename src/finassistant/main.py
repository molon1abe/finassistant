from finassistant.core.logging import setup_logging
from ingest import PdfProcessor

setup_logging()


def main():
    file_to_read = ""
    processor = PdfProcessor()
    processor.open_file(file_to_read)
    print(processor.read_page(0))
    batch = processor.read_batch(100)
    print(batch)
    processor.close_file()


if __name__ == "__main__":
    main()
