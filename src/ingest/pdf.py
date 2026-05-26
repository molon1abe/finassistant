import structlog
from pypdf import PdfReader

logger = structlog.get_logger()


class PdfProcessor:
    def __init__(self):
        pass

    def read_pdf(self, file_name: str) -> str:
        """Read pdf file with pypdf"""
        try:
            reader = PdfReader(file_name)
            logger.info("pdf.parsed", pages=len(reader.pages), file=file_name)
            page = reader.pages[0]
            pdf_text = page.extract_text()
            return pdf_text
        except Exception as e:
            logger.error("ingest.failed", error=str(e))

        return ""
