from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from chain.tokenizer import chunk_by_token, vectorise_docs


def make_docs(*texts: str) -> list[Document]:
    return [Document(page_content=t, metadata={"source": "test"}) for t in texts]


class TestChunkByToken:
    def test_returns_list_of_documents(self):
        docs = make_docs("word " * 100)
        result = chunk_by_token(docs)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_large_document_is_split(self):
        docs = make_docs("word " * 1000)
        result = chunk_by_token(docs, chunk_size=100, chunk_overlap=0)
        assert len(result) > 1

    def test_small_document_is_not_split(self):
        docs = make_docs("short text")
        result = chunk_by_token(docs, chunk_size=800)
        assert len(result) == 1

    def test_overlap_produces_repeated_tokens(self):
        docs = make_docs("word " * 300)
        no_overlap = chunk_by_token(docs, chunk_size=100, chunk_overlap=0)
        with_overlap = chunk_by_token(docs, chunk_size=100, chunk_overlap=50)
        assert len(with_overlap) > len(no_overlap)

    def test_metadata_preserved(self):
        docs = make_docs("word " * 1000)
        result = chunk_by_token(docs, chunk_size=100, chunk_overlap=0)
        assert all(d.metadata.get("source") == "test" for d in result)

    def test_empty_docs_raises(self):
        with pytest.raises(ValueError, match="Cannot chunk empty document list"):
            chunk_by_token([])


class TestVectoriseDocs:
    def test_returns_faiss_vectorstore(self):
        mock_vectorstore = MagicMock()
        with (
            patch("chain.tokenizer.OpenAIEmbeddings"),
            patch("chain.tokenizer.FAISS") as mock_faiss,
        ):
            mock_faiss.from_documents.return_value = mock_vectorstore
            result = vectorise_docs(make_docs("some financial text"))
        assert result is mock_vectorstore

    def test_passes_model_to_embeddings(self):
        with (
            patch("chain.tokenizer.OpenAIEmbeddings") as mock_embeddings,
            patch("chain.tokenizer.FAISS"),
        ):
            vectorise_docs(make_docs("text"), model="text-embedding-3-small")
            mock_embeddings.assert_called_once_with(model="text-embedding-3-small")

    def test_passes_docs_to_faiss(self):
        docs = make_docs("text")
        with (
            patch("chain.tokenizer.OpenAIEmbeddings") as mock_embeddings,
            patch("chain.tokenizer.FAISS") as mock_faiss,
        ):
            vectorise_docs(docs)
            mock_faiss.from_documents.assert_called_once_with(
                documents=docs, embedding=mock_embeddings.return_value
            )

    def test_empty_docs_raises(self):
        with pytest.raises(ValueError, match="Cannot vectorise empty document list"):
            vectorise_docs([])
