from __future__ import annotations

import sqlite3
import sys
import shutil
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    # .../apps/server/ranking/test_hybrid.py -> repo root
    return Path(__file__).resolve().parents[3]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    return v / (np.linalg.norm(v) + 1e-12)


def _print_top(title: str, rows: list[object], limit: int = 5) -> None:
    from ranking.hybrid_ranker import SearchResult

    print()
    print(title)
    print("Top results:")
    for r in rows[:limit]:
        r = r  # type: ignore[no-redef]
        assert isinstance(r, SearchResult)
        print(
            f"- {r.chunk_id:>10}  final={r.score:.4f}  bm25={r.bm25_score:.4f}  vec={r.vec_score:.4f}  "
            f"reasons={r.reasons}"
        )


def main() -> None:
    root = _repo_root()

    # Ensure repo root is importable so `apps.server...` imports work.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Make imports work when running as a script from anywhere.
    server_dir = root / "apps" / "server"
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    from ranking.search import hybrid_search  # noqa: E402
    from vector_index.hnsw_index import HNSWPackIndex  # noqa: E402

    # Use an isolated test database, not the shared dev DB.
    test_db = root / ".localdata" / "test_hybrid.db"
    test_db.parent.mkdir(parents=True, exist_ok=True)
    if test_db.exists():
        test_db.unlink()

    from apps.server.db.migrate import migrate  # noqa: E402

    migrate(
        db_path=test_db,
        schema_sql_path=Path("apps/server/db/schema.sql"),
        migrations_dir=Path("apps/server/db/migrations"),
    )

    print(f"Using test DB: {test_db}")

    con = sqlite3.connect(test_db)
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        pack_id = "pack_hybrid"
        con.execute("INSERT INTO packs(id, name) VALUES (?, ?)", (pack_id, "Hybrid pack"))

        # Case A: synonyms
        # Query: "automobilio mokestis"
        # - literal chunk matches BM25 strongly
        # - synonym chunk should be a better vector match
        a_syn = "a_syn"
        a_lit = "a_lit"

        # Case B: exact quote
        # Query: "\"PVM deklaracija FR0600\""
        # - quote chunk contains exact phrase
        # - semantic chunk is a better vector match, but lacks exact substring order
        b_quote = "b_quote"
        b_sem = "b_sem"

        chunks = [
            (a_syn, pack_id, "Transporto priemonės mokestis: kada taikomas ir kaip apskaičiuojamas."),
            (a_lit, pack_id, "Automobilio mokestis: tarifai ir mokėjimo tvarka."),
            (b_quote, pack_id, "Cituojama: PVM deklaracija FR0600 pateikiama iki mėnesio 25 d."),
            (b_sem, pack_id, "FR0600 PVM deklaracijos pateikimas VMI per EDS: terminai ir žingsniai."),
            ("noise1", pack_id, "Pelno mokesčio deklaracija PLN204."),
            ("noise2", pack_id, "Darbo užmokestis, GPM ir Sodros įmokos."),
        ]

        con.executemany("INSERT INTO chunks(id, pack_id, text) VALUES (?, ?, ?)", chunks)
        con.commit()

        # Synthetic vectors (dim=8), constructed to force the desired ranking behavior.
        dim = 8
        e = np.eye(dim, dtype=np.float32)

        # Query A should match synonym chunk best in vector space.
        qA = _normalize(e[0])
        v_a_syn = _normalize(e[0])  # best
        v_a_lit = _normalize(e[1])  # worse

        # Query B should match semantic chunk best in vector space.
        qB = _normalize(e[2])
        v_b_sem = _normalize(e[2])   # best
        v_b_quote = _normalize(e[3])  # worse

        rng = np.random.default_rng(1)
        vectors_by_id: dict[str, np.ndarray] = {
            a_syn: v_a_syn,
            a_lit: v_a_lit,
            b_quote: v_b_quote,
            b_sem: v_b_sem,
            "noise1": _normalize(rng.normal(size=(dim,)).astype(np.float32)),
            "noise2": _normalize(rng.normal(size=(dim,)).astype(np.float32)),
        }

        # Build HNSW index on disk for this pack.
        pack_dir = root / ".localdata" / "indices" / pack_id
        if pack_dir.exists():
            shutil.rmtree(pack_dir)
        HNSWPackIndex.build(
            pack_id=pack_id,
            dim=dim,
            vectors=vectors_by_id,
            chunk_ids=None,
            out_dir=pack_dir,
            space="cosine",
        )

        # -----------------------
        # Case A: synonyms proof
        # -----------------------
        queryA = "automobilio mokestis"
        bm25_only_A = hybrid_search(
            con,
            pack_id=pack_id,
            query=queryA,
            query_vec=qA,
            fts_k=20,
            vec_k=20,
            out_k=10,
            indices_root=root / ".localdata" / "indices",
            w_bm25=1.0,
            w_vec=0.0,
            quote_boost=0.0,
            quote_miss_penalty=0.0,
            token_boost=0.0,
            auto=False,
        )
        vec_only_A = hybrid_search(
            con,
            pack_id=pack_id,
            query=queryA,
            query_vec=qA,
            fts_k=20,
            vec_k=20,
            out_k=10,
            indices_root=root / ".localdata" / "indices",
            w_bm25=0.0,
            w_vec=1.0,
            quote_boost=0.0,
            quote_miss_penalty=0.0,
            token_boost=0.0,
            auto=False,
        )

        dbgA: dict[str, object] = {}
        resA = hybrid_search(
            con,
            pack_id=pack_id,
            query=queryA,
            query_vec=qA,
            fts_k=20,
            vec_k=20,
            out_k=10,
            indices_root=root / ".localdata" / "indices",
            auto=True,
            debug=dbgA,
        )
        print()
        print("Case A chosen weights:", {k: dbgA[k] for k in ["w_bm25", "w_vec", "quote_boost", "quote_miss_penalty", "token_boost"]})
        _print_top("Case A (synonyms): hybrid_search (auto)", resA)

        if not bm25_only_A or bm25_only_A[0].chunk_id == a_syn:
            raise AssertionError("Setup failed: BM25-only unexpectedly ranked synonym chunk first.")
        if not vec_only_A or vec_only_A[0].chunk_id != a_syn:
            raise AssertionError(f"Setup failed: vector-only expected top1={a_syn}, got={vec_only_A[0].chunk_id if vec_only_A else None}")

        # Required proof:
        # - vector-heavy behavior wins
        # - hybrid top1 == vector-only top1
        # - bm25-only would rank differently
        if not resA or resA[0].chunk_id != vec_only_A[0].chunk_id:
            raise AssertionError(
                f"Case A failed: hybrid top1 != vector-only top1 (hybrid={resA[0].chunk_id if resA else None}, "
                f"vec_only={vec_only_A[0].chunk_id if vec_only_A else None})"
            )
        if resA[0].chunk_id == bm25_only_A[0].chunk_id:
            raise AssertionError("Case A failed: hybrid top1 should differ from bm25-only top1.")

        # -------------------------
        # Case B: quote boost proof
        # -------------------------
        queryB = "\"PVM deklaracija FR0600\""

        # Vector-only ranking should prefer b_sem.
        vec_only = hybrid_search(
            con,
            pack_id=pack_id,
            query=queryB,
            query_vec=qB,
            fts_k=20,
            vec_k=20,
            out_k=10,
            w_bm25=0.0,
            w_vec=1.0,
            quote_boost=0.0,
            quote_miss_penalty=0.0,
            token_boost=0.0,
            indices_root=root / ".localdata" / "indices",
            auto=False,
        )
        _print_top("Case B (exact quote): vector-only ranking", vec_only)
        if not vec_only or vec_only[0].chunk_id != b_sem:
            raise AssertionError(
                f"Setup failed: vector-only expected top1={b_sem}, got={vec_only[0].chunk_id if vec_only else None}"
            )

        bm25_plus_quote = hybrid_search(
            con,
            pack_id=pack_id,
            query=queryB,
            query_vec=qB,
            fts_k=20,
            vec_k=20,
            out_k=10,
            w_bm25=1.0,
            w_vec=0.0,
            quote_boost=0.35,
            quote_miss_penalty=0.15,
            token_boost=0.0,
            indices_root=root / ".localdata" / "indices",
            auto=False,
        )

        dbgB: dict[str, object] = {}
        resB = hybrid_search(
            con,
            pack_id=pack_id,
            query=queryB,
            query_vec=qB,
            fts_k=20,
            vec_k=20,
            out_k=10,
            indices_root=root / ".localdata" / "indices",
            auto=True,
            debug=dbgB,
        )
        print()
        print("Case B chosen weights:", {k: dbgB[k] for k in ["w_bm25", "w_vec", "quote_boost", "quote_miss_penalty", "token_boost"]})
        _print_top("Case B (exact quote): bm25+quote", bm25_plus_quote)
        _print_top("Case B (exact quote): hybrid_search (auto)", resB)
        if not resB or resB[0].chunk_id != b_quote:
            raise AssertionError(f"Case B failed: expected top1={b_quote}, got={resB[0].chunk_id if resB else None}")
        if bm25_plus_quote[0].chunk_id != b_quote:
            raise AssertionError("Setup failed: bm25+quote did not rank quoted chunk first.")

        # Required proof:
        # - quoted chunk must rank #1
        # - hybrid top1 == bm25+quote
        # - vector-only would rank a different chunk higher
        if resB[0].chunk_id != bm25_plus_quote[0].chunk_id:
            raise AssertionError("Case B failed: hybrid top1 != bm25+quote top1.")
        if vec_only[0].chunk_id == resB[0].chunk_id:
            raise AssertionError("Case B failed: vector-only should rank a different chunk higher than hybrid.")

        print()
        print("Hybrid B4 test OK")
        print(f"- db={test_db}")
        print(f"- pack_id={pack_id}")
        print(f"- index_dir={pack_dir}")

    finally:
        con.close()


if __name__ == "__main__":
    main()

