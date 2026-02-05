export type CaseOut = {
  case_id: string;
  title: string;
  description?: string | null;
  category?: string | null;
  created_at_utc: string;
};

export type DocumentOut = {
  id: number | string; // server id or local temp id
  case_id: string;
  original_name: string;
  mime?: string;
  size_bytes: number;
  uploaded_at_utc?: string;
  status: "uploading" | "queued" | "processing" | "done" | "failed";
  error?: string | null;
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${txt || res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function createCase(payload: {
  title: string;
  description?: string;
  category?: string;
}): Promise<CaseOut> {
  return apiFetch<CaseOut>("/api/cases", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getCase(caseId: string): Promise<CaseOut> {
  return apiFetch<CaseOut>(`/api/cases/${encodeURIComponent(caseId)}`);
}

export async function uploadDocuments(caseId: string, files: File[]): Promise<DocumentOut[]> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);

  return apiFetch<DocumentOut[]>(`/api/cases/${encodeURIComponent(caseId)}/documents`, {
    method: "POST",
    body: fd,
  });
}

export async function listDocuments(caseId: string): Promise<DocumentOut[]> {
  const r = await apiFetch<{ documents: DocumentOut[] }>(
    `/api/cases/${encodeURIComponent(caseId)}/documents`
  );
  return r.documents;
}

