# Offline LLM (llama.cpp) runtime

This server can call a locally bundled `llama.cpp` CLI binary **offline** (no internet).

## Environment variables

- `LEX_LLAMA_BIN`: full path to the `llama-cli` (or `main`) executable.
- `LEX_MODEL_GGUF`: full path to a `.gguf` model file.

If `LEX_LLAMA_BIN` is not set, the runtime looks for:

- `<data_dir>/bin/llama-cli(.exe)` or `<app_dir>/bin/llama-cli(.exe)`
- `<data_dir>/bin/main(.exe)` or `<app_dir>/bin/main(.exe)`

If `LEX_MODEL_GGUF` is not set, the runtime uses **the single** `*.gguf` file under `model_dir/`.

## Example (Windows PowerShell)

```powershell
$env:LEX_LLAMA_BIN = "C:\path\to\llama-cli.exe"
$env:LEX_MODEL_GGUF = "C:\path\to\model.gguf"
```

Then from `apps/server`:

```powershell
python -m pytest -q
```

The optional integration smoke test will run only if both env vars are set.

