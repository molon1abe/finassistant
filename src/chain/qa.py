from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough


def build_qa_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    def format_context(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(
        "Answer in the same language as the question. Use only the provided context.\nContext:\n{context}\n\nQuestion: {question}"
    )
    return (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(chain: Runnable, question: str) -> str:
    return chain.invoke(question)
