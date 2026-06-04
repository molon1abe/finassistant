from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from chain.chunker import chunk_by_char, chunk_by_token
from chain.store import DEFAULT_DB_PATH, build_store, load_store, store_exists


def make_docs(*texts: str) -> list[Document]:
    return [Document(page_content=t, metadata={"source": "test"}) for t in texts]


class TestChunkByToken:
    def test_returns_list_of_documents(self):
        result = chunk_by_token(make_docs("word " * 100))
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_large_document_is_split(self):
        result = chunk_by_token(
            make_docs("word " * 1000), chunk_size=100, chunk_overlap=0
        )
        assert len(result) > 1

    def test_small_document_is_not_split(self):
        result = chunk_by_token(make_docs("short text"), chunk_size=800)
        assert len(result) == 1

    def test_overlap_produces_more_chunks(self):
        docs = make_docs("word " * 300)
        no_overlap = chunk_by_token(docs, chunk_size=100, chunk_overlap=0)
        with_overlap = chunk_by_token(docs, chunk_size=100, chunk_overlap=50)
        assert len(with_overlap) > len(no_overlap)

    def test_metadata_preserved(self):
        result = chunk_by_token(
            make_docs("word " * 1000), chunk_size=100, chunk_overlap=0
        )
        assert all(d.metadata.get("source") == "test" for d in result)

    def test_empty_docs_raises(self):
        with pytest.raises(ValueError, match="Cannot chunk empty document list"):
            chunk_by_token([])


class TestChunkByChar:
    def test_returns_list_of_documents(self):
        result = chunk_by_char(make_docs("word " * 100))
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_large_document_is_split(self):
        result = chunk_by_char(
            make_docs("word " * 500), chunk_size=100, chunk_overlap=0
        )
        assert len(result) > 1

    def test_small_document_is_not_split(self):
        result = chunk_by_char(make_docs("short text"), chunk_size=1000)
        assert len(result) == 1

    def test_overlap_produces_more_chunks(self):
        docs = make_docs("word " * 300)
        no_overlap = chunk_by_char(docs, chunk_size=100, chunk_overlap=0)
        with_overlap = chunk_by_char(docs, chunk_size=100, chunk_overlap=50)
        assert len(with_overlap) > len(no_overlap)

    def test_metadata_preserved(self):
        result = chunk_by_char(
            make_docs("word " * 500), chunk_size=100, chunk_overlap=0
        )
        assert all(d.metadata.get("source") == "test" for d in result)

    def test_empty_docs_raises(self):
        with pytest.raises(ValueError, match="Cannot chunk empty document list"):
            chunk_by_char([])


class TestBuildStore:
    def test_returns_vectorstore(self):
        mock_store = MagicMock()
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = mock_store
            result = build_store(make_docs("text"), MagicMock(), collection_name="test")
        assert result is mock_store

    def test_adds_documents(self):
        mock_store = MagicMock()
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = mock_store
            build_store(make_docs("text"), MagicMock(), collection_name="test")
        mock_store.add_documents.assert_called_once()

    def test_uses_default_path(self):
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            build_store(make_docs("text"), MagicMock(), collection_name="test")
            _, kwargs = mock_chroma.call_args
            assert kwargs["persist_directory"] == str(DEFAULT_DB_PATH)

    def test_accepts_custom_path(self):
        custom_path = "/tmp/test_db"
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            build_store(
                make_docs("text"), MagicMock(), collection_name="test", path=custom_path
            )
            _, kwargs = mock_chroma.call_args
            assert kwargs["persist_directory"] == custom_path

    def test_passes_collection_name(self):
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            build_store(make_docs("text"), MagicMock(), collection_name="my_col")
            _, kwargs = mock_chroma.call_args
            assert kwargs["collection_name"] == "my_col"

    def test_empty_docs_raises(self):
        with pytest.raises(
            ValueError, match="Cannot build store from empty document list"
        ):
            build_store([], MagicMock(), collection_name="test")


class TestLoadStore:
    def test_returns_vectorstore(self):
        mock_store = MagicMock()
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = mock_store
            result = load_store(MagicMock(), collection_name="test")
        assert result is mock_store

    def test_uses_default_path(self):
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            load_store(MagicMock(), collection_name="test")
            _, kwargs = mock_chroma.call_args
            assert kwargs["persist_directory"] == str(DEFAULT_DB_PATH)

    def test_passes_collection_name(self):
        with patch("chain.store.Chroma") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            load_store(MagicMock(), collection_name="my_col")
            _, kwargs = mock_chroma.call_args
            assert kwargs["collection_name"] == "my_col"


class TestStoreExists:
    def test_returns_true_when_path_exists(self, tmp_path):
        assert store_exists(str(tmp_path)) is True

    def test_returns_false_when_path_missing(self):
        assert store_exists("/nonexistent/path/db") is False
