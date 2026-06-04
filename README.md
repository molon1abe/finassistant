# finassistant

A local-first chatbot that ingests bank statements (CSV/PDF) and answers natural language questions about your spending. All data stays on your machine — no cloud services required.

## Features

- Parses **Sberbank debit card statements** (PDF) and generic PDFs/CSVs
- Stores document chunks in a local **ChromaDB** vector database
- Multi-turn conversation with **in-memory chat history** per session
- Runs entirely offline via **Ollama** (default: `aya-expanse:8b` + `nomic-embed-text`)
- Optional OpenAI backend for cloud mode
- Two interfaces: **CLI** and **Gradio web UI**

## Requirements

- Python 3.14+
- [Ollama](https://ollama.com) running locally (for local mode)

```bash
ollama pull aya-expanse:8b
ollama pull nomic-embed-text
```

## Installation

```bash
git clone <repo>
cd finassistant
uv sync
```

## Configuration

Settings are read from a `.env` file in the project root. All fields are optional.

```dotenv
# Local mode (default)
MODEL='{"type":"local","model_name":"aya-expanse:8b","embed_model":"nomic-embed-text","base_url":"http://localhost:11434"}'

# Cloud mode (OpenAI)
# MODEL='{"type":"cloud","openai_api_key":"<your-openai-key>","model_name":"gpt-4o-mini"}'

DB_PATH=db
LOG_LEVEL=INFO
```

## Usage

### CLI

```bash
# Ingest a bank statement
uv run python -m finassistant.cli ingest path/to/statement.pdf

# Ask a question
uv run python -m finassistant.cli ask-question "How much did I spend on groceries?"

# Show database stats
uv run python -m finassistant.cli status

# Inspect stored chunks
uv run python -m finassistant.cli peek --n 10
```

### Web UI (Gradio)

```bash
uv run python -m finassistant.ui
```

Open `http://localhost:7860` in your browser. Three tabs are available:

- **Ask** — ask questions about ingested documents (conversation history is kept for the session)
- **Ingest** — load a new PDF or CSV file by path
- **Status** — show the current state of the vector database

## Supported file formats

| Format | Notes |
|--------|-------|
| Sberbank PDF | Debit card statements (`Выписка по счету дебетовой карты`). Detected automatically. |
| Generic PDF | Parsed page-by-page as fallback when Sberbank format is not detected. |
| CSV | Any CSV with a header row; each row becomes one document chunk. |

