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


@app.command()
def peek(
    ctx: typer.Context,
    n: int = typer.Option(5, "--n", help="Number of chunks to show"),
):
    """Print stored chunks to verify translation and vectorization."""
    store = ctx.obj["store"]
    results = store.get()
    ids = results["ids"]
    documents = results["documents"]
    if not ids:
        typer.echo("Store is empty.")
        return
    typer.echo(f"Showing {min(n, len(ids))} of {len(ids)} chunks:\n")
    for i, (chunk_id, content) in enumerate(zip(ids[:n], documents[:n])):
        typer.echo(f"[{i + 1}] id={chunk_id}")
        typer.echo(content[:300])
        typer.echo("---")


def main():
    app()


if __name__ == "__main__":
    main()
