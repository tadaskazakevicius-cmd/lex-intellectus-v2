export function App() {
  return (
    <div className="page">
      <header className="header">
        <div className="brand">Lex Intellectus</div>
        <div className="muted">Project base (A1–A3)</div>
      </header>

      <main className="card">
        <h1>It works.</h1>
        <p className="muted">
          This SPA is served from the FastAPI server with an <code>index.html</code> fallback for
          client-side routing.
        </p>

        <ul>
          <li>
            <span className="label">Health:</span> <code>/api/health</code>
          </li>
        </ul>
      </main>
    </div>
  );
}

