from __future__ import annotations
import sqlite3
from pathlib import Path

DB = Path(".localdata") / "app.db"

def main() -> None:
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON;")

    # Clean slate for repeatable run (MVP)
    con.execute("DELETE FROM chunks;")
    con.execute("DELETE FROM packs;")
    con.commit()

    # Seed one pack
    con.execute("INSERT INTO packs(id, name) VALUES (?, ?)", ("pack1", "Test pack"))

    # 10 chunks with varying relevance to query "pvm deklaracija"
    chunks = [
        ("c1", "pack1", "PVM deklaracija FR0600 pildymas ir pateikimas VMI."),
        ("c2", "pack1", "Kaip pildyti deklaraciją dėl PVM (FR0600) ir terminai."),
        ("c3", "pack1", "PVM mokėtojo registracija ir prievolės."),
        ("c4", "pack1", "Pelno mokesčio deklaracija PLN204."),
        ("c5", "pack1", "PVM sąskaita faktūra: privalomi rekvizitai."),
        ("c6", "pack1", "Darbo užmokestis, GPM ir Sodros įmokos."),
        ("c7", "pack1", "Pridėtinės vertės mokestis: lengvatos ir tarifai."),
        ("c8", "pack1", "Deklaracijų pateikimas per EDS: žingsniai."),
        ("c9", "pack1", "FR0600: PVM deklaracija – dažniausios klaidos."),
        ("c10","pack1", "Citata: 'PVM deklaracija FR0600' pateikiama iki mėnesio 25 d."),
    ]

    con.executemany(
        "INSERT INTO chunks(id, pack_id, text) VALUES (?, ?, ?)",
        chunks
    )
    con.commit()

    query = "pvm deklaracija fr0600"

    # Verify hits + bm25 ordering (lower bm25 => better match)
    rows = con.execute(
        """
        SELECT
          chunks.id,
          bm25(chunks_fts) AS score,
          substr(chunks.text, 1, 80) AS snippet
        FROM chunks_fts
        JOIN chunks ON chunks_fts.rowid = chunks.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score ASC
        LIMIT 10;
        """,
        (query,)
    ).fetchall()

    print(f"Query: {query}")
    print("Top hits (best first):")
    for cid, score, snip in rows:
        print(f"- {cid:>3}  bm25={score:.4f}  {snip}")

    if len(rows) == 0:
        raise SystemExit("No hits returned — FTS sync is broken.")

    # Basic expectation: the ones mentioning exact 'FR0600' should rank high
    top_ids = [r[0] for r in rows[:3]]
    print("Top3 IDs:", top_ids)

    con.close()

if __name__ == "__main__":
    main()

