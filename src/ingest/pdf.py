from typing import Generator
import structlog
from pypdf import PdfReader
from pypdf.errors import PdfReadError, EmptyFileError
from itertools import islice

logger = structlog.get_logger()


class PdfProcessor:
    """
    Pdf files processor allows read pdf files in different modes.
    Using pypdf as main tool for reading
    """

    def __init__(self):
        self._reader = None
        self._file_name = ""

    def open_file(self, file_name: str) -> None:
        """Open file for processing"""
        if self._reader is not None:
            self._reader.close()

        try:
            self._reader = PdfReader(file_name)
            self._file_name = file_name
            logger.info("pdf.opened", file=file_name)
        except (FileNotFoundError, PdfReadError, EmptyFileError, IsADirectoryError):
            logger.error("pdf.failed: failed to open file", file=file_name)

    def close_file(self) -> None:
        """Close file by processing end"""
        if self._reader is None:
            logger.error("ingest.failed: Error while closing reader of 'None' value")
            raise ValueError("Error while closing raader of 'None' value")
        self._reader.close()
        self._reader = None
        logger.info("pdf.closed", file=self._file_name)
        self._file_name = ""

    def read_page(self, page_number: int) -> str:
        """Read one specific page from document"""
        try:
            page = self._reader.pages[page_number]
            logger.debug("ingest.debug: successfully loaded page", page_num=page_number)
            return page.extract_text() or ""
        except IndexError as e:
            logger.error("ingest.failed", error=str(e))
        except AttributeError as e:
            logger.error("ingest.failed", error=str(e))
        return ""

    def read_batch(self, batch_size: int) -> Generator[list[dict], None, None]:
        """
        Read document content by batches.
        Batch size expressed in pages
        """
        if not self._reader:
            raise ValueError("Error reading file, reader obj is None")
        pages = enumerate(self._reader.pages)
        it = iter(pages)
        while True:
            chunk = list(islice(it, batch_size))
            if not chunk:
                break
            yield [
                {"page": num, "text": page.extract_text() or ""} for num, page in chunk
            ]

    def __del__(self):
        if self._reader is not None:
            self._reader.close()
