from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnablePassthrough


def build_qa_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_template(
        "Answer using this context:\n{context}\n\nQuestion: {question}"
    )
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(chain: Runnable, question: str) -> str:
    return chain.invoke(question)
