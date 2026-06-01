import typer
from finassistant.core import Settings
from chain import ask
from finassistant.service import (
    init_store,
    add_document,
    init_model,
    build_embeddings,
    build_chain,
)


app = typer.Typer(help="Financial document assistant")


@app.callback()
def setup(ctx: typer.Context):
    ctx.ensure_object(dict)
    settings = Settings()
    embeddings = build_embeddings(settings)
    store = init_store(settings, embeddings)
    ctx.obj["settings"] = settings
    ctx.obj["store"] = store


@app.command()
def ingest(
    ctx: typer.Context,
    file_path: str = typer.Argument(..., help="Path to PDF or CSV file"),
):
    """Ingest a document into the vector store."""
    add_document(ctx.obj["settings"], ctx.obj["store"], file_path)
    typer.echo(f"Ingested: {file_path}")


@app.command()
def ask_question(
    ctx: typer.Context, question: str = typer.Argument(..., help="Question to ask")
):
    """Ask a question about ingested documents."""
    settings, store = ctx.obj["settings"], ctx.obj["store"]
    llm = init_model(settings)
    chain = build_chain(store, llm)
    answer = ask(chain, question)
    typer.echo(answer)


@app.command()
def status(ctx: typer.Context):
    """Show store info."""
    settings, store = ctx.obj["settings"], ctx.obj["store"]
    results = store.get()
    typer.echo("Collection: statements")
    typer.echo(f"Documents: {len(results['ids'])}")
    typer.echo(f"Store path: {settings.db_path}")


def main():
    app()


if __name__ == "__main__":
    main()
