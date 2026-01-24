## Verification

Each task below includes a minimal **How to verify** section.

### A1) Repo scaffold + coding standards

#### How to verify

- **Python compile**:

```bash
cd apps/server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
set PYTHONPATH=src
python -m compileall src
```

- **Server starts on localhost**:

```bash
cd apps/server
set PYTHONPATH=src
python -m uvicorn lex_server.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/api/health`.

---

### A2) Config and OS-agnostic data dirs

#### How to verify

- **First start creates dirs and writes audit entry** (using local workspace dirs via env overrides):

```bash
cd apps/server
set PYTHONPATH=src
set LEX_APP_DIR=..\..
set LEX_DATA_DIR=..\..\.localdata
set LEX_MODEL_DIR=..\..\.localdata\models
set LEX_TEMP_DIR=..\..\.localtemp
python -m uvicorn lex_server.main:app --host 127.0.0.1 --port 8001
```

Then confirm:

- `..\..\.localdata\audit_log.jsonl` exists
- Its first line is JSON containing **event_type**, **timestamp_utc**, and **app_version**

---

### A3) SPA build → embedded assets served by server

#### How to verify

- **Build UI and embed into server**:

```bash
cd apps/ui
npm install
npm run build
```

- **Serve SPA from FastAPI (no external resources)**:

```bash
cd apps/server
set PYTHONPATH=src
python -m uvicorn lex_server.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` and confirm the app loads.

To verify there are no external resources:

- In browser DevTools → Network tab, reload and confirm all requests are to `127.0.0.1`.

