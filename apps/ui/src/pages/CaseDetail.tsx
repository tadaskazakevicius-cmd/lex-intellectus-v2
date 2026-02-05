import { useEffect, useMemo, useRef, useState } from "react";
import type { CaseOut, DocumentOut } from "../api";
import { getCase, listDocuments, uploadDocuments } from "../api";

function isTerminal(s: DocumentOut["status"]) {
  return s === "done" || s === "failed";
}

function formatBytes(n: number) {
  if (!Number.isFinite(n)) return "";
  if (n < 1024) return `${n} B`;
  const kb = n / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

export function CaseDetailPage({ caseId }: { caseId: string }) {
  const [caseInfo, setCaseInfo] = useState<CaseOut | null>(null);
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const pollTimer = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const hasNonTerminal = useMemo(() => docs.some((d) => !isTerminal(d.status) && d.status !== "uploading"), [docs]);

  async function refreshDocuments() {
    const serverDocs = await listDocuments(caseId);
    setDocs((prev) => {
      // Keep local uploading rows; replace/merge server rows by id.
      const uploading = prev.filter((d) => d.status === "uploading");
      return [...uploading, ...serverDocs];
    });
  }

  useEffect(() => {
    setError(null);
    setUploadErr(null);
    setCaseInfo(null);
    setDocs([]);

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    (async () => {
      try {
        const c = await getCase(caseId);
        setCaseInfo(c);
        await refreshDocuments();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    })();

    return () => {
      ac.abort();
      if (pollTimer.current) {
        window.clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId]);

  useEffect(() => {
    // Start polling when there are non-terminal docs; stop when all terminal.
    if (!hasNonTerminal) {
      if (pollTimer.current) {
        window.clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
      return;
    }

    if (pollTimer.current) return;
    pollTimer.current = window.setInterval(() => {
      refreshDocuments().catch(() => {
        // keep polling; show error inline once
      });
    }, 2000);

    return () => {
      if (pollTimer.current) {
        window.clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasNonTerminal, caseId]);

  async function onPickFiles(fileList: FileList | null) {
    setUploadErr(null);
    if (!fileList || fileList.length === 0) return;
    const files = Array.from(fileList);

    const localRows: DocumentOut[] = files.map((f, idx) => ({
      id: `local-${Date.now()}-${idx}`,
      case_id: caseId,
      original_name: f.name,
      size_bytes: f.size,
      status: "uploading",
      error: null,
    }));
    setDocs((prev) => [...localRows, ...prev]);

    setBusy(true);
    try {
      const uploaded = await uploadDocuments(caseId, files);
      setDocs((prev) => {
        const keepUploading = prev.filter((d) => d.status === "uploading");
        return [...keepUploading, ...uploaded, ...prev.filter((d) => d.status !== "uploading")];
      });
      await refreshDocuments();
    } catch (err) {
      setUploadErr(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      // remove uploading placeholders (either replaced by refresh or after error)
      setDocs((prev) => prev.filter((d) => d.status !== "uploading"));
    }
  }

  return (
    <main className="card">
      <div className="row">
        <div>
          <h1>Byla</h1>
          <div className="muted">
            ID: <code>{caseId}</code>
          </div>
          {caseInfo ? (
            <div className="muted">
              <div>
                <span className="label">Pavadinimas:</span> {caseInfo.title}
              </div>
              {caseInfo.category ? (
                <div>
                  <span className="label">Kategorija:</span> {caseInfo.category}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      {error ? <div className="error">{error}</div> : null}

      <section className="panel">
        <h2>Įkelti dokumentus</h2>
        <div className="muted">PDF / DOCX / TXT. Galima pasirinkti kelis failus.</div>
        <div className="actions">
          <input
            type="file"
            multiple
            onChange={(e) => onPickFiles(e.target.files)}
            disabled={busy}
          />
        </div>
        {uploadErr ? <div className="error">{uploadErr}</div> : null}
      </section>

      <section className="panel">
        <h2>Dokumentai</h2>
        <div className="tableWrap">
          <table className="table">
            <thead>
              <tr>
                <th>Pavadinimas</th>
                <th>Dydis</th>
                <th>Įkelta</th>
                <th>Statusas</th>
                <th>Klaida</th>
              </tr>
            </thead>
            <tbody>
              {docs.length === 0 ? (
                <tr>
                  <td colSpan={5} className="muted">
                    Dokumentų dar nėra.
                  </td>
                </tr>
              ) : (
                docs.map((d) => (
                  <tr key={String(d.id)}>
                    <td>{d.original_name}</td>
                    <td className="muted">{formatBytes(d.size_bytes)}</td>
                    <td className="muted">{d.uploaded_at_utc || "-"}</td>
                    <td>
                      <span className={`pill pill-${d.status}`}>{d.status}</span>
                    </td>
                    <td className="muted">{d.error || ""}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

