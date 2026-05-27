from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from ingest import load_csv, load_pdf

FIXTURES_DIR = Path(__file__).parent / "data"
CSV_FILE = FIXTURES_DIR / "bank_statement.csv"


class TestLoadCsv:
    def test_returns_list_of_documents(self):
        docs = load_csv(str(CSV_FILE))
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_row_count(self):
        docs = load_csv(str(CSV_FILE))
        assert len(docs) == 22  # 22 data rows, header excluded

    def test_documents_have_content(self):
        docs = load_csv(str(CSV_FILE))
        assert all(d.page_content for d in docs)

    def test_metadata_has_source(self):
        docs = load_csv(str(CSV_FILE))
        assert all("source" in d.metadata for d in docs)

    def test_file_not_found(self):
        with pytest.raises(Exception):
            load_csv("nonexistent.csv")


class TestLoadPdf:
    def test_returns_list_of_documents(self):
        mock_docs = [
            Document(
                page_content="bank statement text",
                metadata={"source": "test.pdf", "page": 0},
            )
        ]
        with patch("ingest.pdf.PyMuPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = mock_docs
            result = load_pdf("test.pdf")
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_returns_one_document_per_page(self):
        mock_docs = [
            Document(page_content="page 1", metadata={"source": "test.pdf", "page": 0}),
            Document(page_content="page 2", metadata={"source": "test.pdf", "page": 1}),
        ]
        with patch("ingest.pdf.PyMuPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = mock_docs
            result = load_pdf("test.pdf")
        assert len(result) == 2
        assert result[0].page_content == "page 1"
        assert result[1].metadata["page"] == 1

    def test_file_not_found(self):
        with pytest.raises(Exception):
            load_pdf("nonexistent.pdf")
