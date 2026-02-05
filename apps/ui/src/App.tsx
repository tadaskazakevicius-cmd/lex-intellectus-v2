import { CaseDetailPage } from "./pages/CaseDetail";
import { NewCasePage } from "./pages/NewCase";
import { navigate, useRoute } from "./router";

export function App() {
  const route = useRoute();

  return (
    <div className="page">
      <header className="header">
        <div className="brand" role="button" tabIndex={0} onClick={() => navigate("/")}>
          Lex Intellectus
        </div>
        <div className="muted">
          <a className="link" href="/cases/new" onClick={(e) => (e.preventDefault(), navigate("/cases/new"))}>
            Nauja byla
          </a>
        </div>
      </header>

      {route.name === "cases_new" ? <NewCasePage /> : null}
      {route.name === "case_detail" ? <CaseDetailPage caseId={route.caseId} /> : null}
      {route.name === "home" ? (
        <main className="card">
          <h1>Pradžia</h1>
          <p className="muted">MVP: sukurk bylą ir įkelk dokumentus.</p>
          <div className="actions">
            <button className="btn" onClick={() => navigate("/cases/new")}>
              Nauja byla
            </button>
          </div>
        </main>
      ) : null}
    </div>
  );
}

