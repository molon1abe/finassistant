from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from chain.tokenizer import DEFAULT_DB_PATH, chunk_by_token, vectorise_chroma


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


class TestVectoriseChroma:
    def test_returns_vectorstore(self):
        mock_store = MagicMock()
        with patch("chain.tokenizer.Chroma") as mock_chroma:
            mock_chroma.from_documents.return_value = mock_store
            result = vectorise_chroma(make_docs("text"), embeddings=MagicMock())
        assert result is mock_store

    def test_uses_default_persist_directory(self):
        with patch("chain.tokenizer.Chroma") as mock_chroma:
            vectorise_chroma(make_docs("text"), embeddings=MagicMock())
            _, kwargs = mock_chroma.from_documents.call_args
            assert kwargs["persist_directory"] == DEFAULT_DB_PATH

    def test_accepts_custom_persist_directory(self):
        custom_path = Path("/tmp/test_db")
        with patch("chain.tokenizer.Chroma") as mock_chroma:
            vectorise_chroma(
                make_docs("text"), embeddings=MagicMock(), persist_directory=custom_path
            )
            _, kwargs = mock_chroma.from_documents.call_args
            assert kwargs["persist_directory"] == custom_path

    def test_passes_embeddings_to_chroma(self):
        mock_embeddings = MagicMock()
        with patch("chain.tokenizer.Chroma") as mock_chroma:
            vectorise_chroma(make_docs("text"), embeddings=mock_embeddings)
            args, _ = mock_chroma.from_documents.call_args
            assert args[1] is mock_embeddings

    def test_empty_docs_raises(self):
        with pytest.raises(ValueError, match="Cannot vectorise empty document list"):
            vectorise_chroma([], embeddings=MagicMock())
