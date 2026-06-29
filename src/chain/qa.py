from collections.abc import AsyncIterator

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory

_DEFAULT_SESSION = "default"


def build_qa_chain(
    retriever: BaseRetriever, llm: BaseChatModel
) -> RunnableWithMessageHistory:
    def format_context(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer in the same language as the question. "
                "Use only the provided context.\nContext:\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    core_chain = (
        {
            "context": (lambda x: x["question"])
            | retriever
            | RunnableLambda(format_context),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


def ask(
    chain: RunnableWithMessageHistory, question: str, session_id: str = _DEFAULT_SESSION
) -> str:
    return chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )


async def ask_stream(
    chain: RunnableWithMessageHistory,
    question: str,
    session_id: str = _DEFAULT_SESSION,
) -> AsyncIterator[str]:
    async for chunk in chain.astream(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk
