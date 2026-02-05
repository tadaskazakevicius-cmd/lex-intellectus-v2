import { useState } from "react";
import { createCase } from "../api";
import { navigate } from "../router";

export function NewCasePage() {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!title.trim()) {
      setError("Pavadinimas privalomas.");
      return;
    }

    setBusy(true);
    try {
      const c = await createCase({
        title: title.trim(),
        description: description.trim() || undefined,
        category: category.trim() || undefined,
      });
      navigate(`/cases/${encodeURIComponent(c.case_id)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="card">
      <h1>Nauja byla</h1>
      <p className="muted">Sukurk bylą ir įkelk dokumentus.</p>

      <form className="form" onSubmit={onSubmit}>
        <label className="field">
          <div className="fieldLabel">Pavadinimas *</div>
          <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Bylos pavadinimas" />
        </label>

        <label className="field">
          <div className="fieldLabel">Aprašymas</div>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Trumpas aprašymas (nebūtina)"
            rows={3}
          />
        </label>

        <label className="field">
          <div className="fieldLabel">Kategorija</div>
          <input value={category} onChange={(e) => setCategory(e.target.value)} placeholder="Pvz. Baudžiamoji" />
        </label>

        {error ? <div className="error">{error}</div> : null}

        <div className="actions">
          <button className="btn" type="submit" disabled={busy}>
            {busy ? "Kuriama..." : "Sukurti bylą"}
          </button>
        </div>
      </form>
    </main>
  );
}

