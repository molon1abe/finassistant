from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingest import load_csv, load_pdf, load_sber_pdf

FIXTURES_DIR = Path(__file__).parent / "data"
CSV_FILE = FIXTURES_DIR / "bank_statement.csv"
PDF_FILE = FIXTURES_DIR / "bank_statement.pdf"
SBER_PDF_FILE = FIXTURES_DIR / "bank_statement.pdf"


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
        result = load_pdf(str(PDF_FILE))
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_returns_one_document_per_page(self):
        result = load_pdf(str(PDF_FILE))
        assert len(result) == 2

    def test_documents_have_content(self):
        result = load_pdf(str(PDF_FILE))
        assert all(d.page_content for d in result)

    def test_metadata_has_source(self):
        result = load_pdf(str(PDF_FILE))
        assert all("source" in d.metadata for d in result)

    def test_file_not_found(self):
        with pytest.raises(Exception):
            load_pdf("nonexistent.pdf")


class TestLoadSberPdf:
    def test_returns_list(self):
        result = load_sber_pdf(str(SBER_PDF_FILE))
        assert isinstance(result, list)

    def test_documents_are_document_instances(self):
        result = load_sber_pdf(str(SBER_PDF_FILE))
        assert all(isinstance(d, Document) for d in result)

    def test_transaction_documents_have_content(self):
        result = load_sber_pdf(str(SBER_PDF_FILE))
        if result:  # may be empty if file is not a Sber statement
            assert all(d.page_content for d in result)

    def test_metadata_has_file_id_and_source(self):
        result = load_sber_pdf(str(SBER_PDF_FILE))
        for doc in result:
            assert "file_id" in doc.metadata
            assert "source" in doc.metadata

    def test_transaction_labels_present(self):
        result = load_sber_pdf(str(SBER_PDF_FILE))
        for doc in result:
            assert "Transaction Date:" in doc.page_content
            assert "Amount (RUB):" in doc.page_content

    def test_file_not_found(self):
        with pytest.raises(Exception):
            load_sber_pdf("nonexistent.pdf")
