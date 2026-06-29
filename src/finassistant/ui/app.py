import gradio as gr
from finassistant.core import Settings
from finassistant.service import (
    add_document,
    build_chain,
    build_embeddings,
    init_model,
    init_store,
)
from chain import ask_stream


class FinAssistant:
    def __init__(self):
        settings = Settings()
        embeddings = build_embeddings(settings)
        store = init_store(settings, embeddings)
        self.settings = settings
        self.store = store
        self.chain = build_chain(store, init_model(settings))

    def ingest(self, file_path: str) -> str:
        if not file_path:
            return "No file path provided."
        add_document(self.settings, self.store, file_path)
        return f"Ingested: {file_path}"

    async def ask(self, question: str):
        if not question:
            yield "No question provided."
            return
        full = ""
        async for chunk in ask_stream(self.chain, question):
            full += chunk
            yield full

    def status(self) -> str:
        results = self.store.get()
        return (
            f"Collection: statements\n"
            f"Documents: {len(results['ids'])}\n"
            f"Store path: {self.settings.db_path}"
        )


def build_ui() -> gr.Blocks:
    assistant = FinAssistant()

    with gr.Blocks(title="Financial Assistant") as ui:
        with gr.Tab("Ask"):
            question = gr.Textbox(
                label="Question", placeholder="What did I spend on groceries?"
            )
            answer = gr.Textbox(label="Answer", interactive=False, lines=5)
            gr.Button("Ask").click(assistant.ask, inputs=question, outputs=answer)

        with gr.Tab("Ingest"):
            file_path = gr.Textbox(
                label="File path", placeholder="tests/data/bank_statement.pdf"
            )
            result = gr.Textbox(label="Result", interactive=False)
            gr.Button("Ingest").click(
                assistant.ingest, inputs=file_path, outputs=result
            )

        with gr.Tab("Status"):
            info = gr.Textbox(label="Store info", interactive=False, lines=4)
            gr.Button("Refresh").click(assistant.status, inputs=None, outputs=info)

    return ui


def main():
    build_ui().launch(max_threads=4)


if __name__ == "__main__":
    main()
