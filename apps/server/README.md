# Server (FastAPI)

Minimal FastAPI server scaffold for Lex Intellectus.

## Dev setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Run

```bash
$env:PYTHONPATH = "src"
python -m uvicorn lex_server.main:app --host 127.0.0.1 --port 8000
```

## Lint / format / test

```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```
