from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, cast

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "numpy is required for vector indexing. Install server deps (see requirements.txt)."
    ) from e

try:
    import hnswlib
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "hnswlib is required for vector indexing. Install server deps (see requirements.txt)."
    ) from e


B3_SCHEMA_VERSION = 1


@dataclass
class HNSWPackIndex:
    pack_id: str
    dim: int
    space: str
    index: "hnswlib.Index"
    label_to_chunk_id: dict[int, str]
    chunk_id_to_label: dict[str, int]

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _ensure_float32_2d(a: "np.ndarray", dim: int) -> "np.ndarray":
        arr = np.asarray(a, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"Expected array shape (N,{dim}); got {arr.shape}")
        return arr

    @classmethod
    def build(
        cls,
        pack_id: str,
        dim: int,
        vectors: Mapping[str, Sequence[float] | "np.ndarray"] | "np.ndarray",
        chunk_ids: Sequence[str] | None,
        out_dir: Path,
        *,
        space: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
    ) -> "HNSWPackIndex":
        """
        Build and persist an on-disk HNSW index for one pack.

        Determinism:
        - Labels are assigned 0..N-1 in sorted(chunk_id) order.

        Files written under `out_dir`:
        - hnsw.bin
        - idmap.json
        - meta.json
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(vectors, np.ndarray):
            if chunk_ids is None:
                raise ValueError("chunk_ids is required when vectors is a numpy array")
            raw_chunk_ids = list(chunk_ids)
            data_by_id = {cid: vectors[i] for i, cid in enumerate(raw_chunk_ids)}
        else:
            data_by_id = dict(vectors)

        ordered_chunk_ids = sorted(data_by_id.keys())
        n = len(ordered_chunk_ids)
        if n == 0:
            raise ValueError("Cannot build index with 0 vectors")

        labels = np.arange(n, dtype=np.int32)
        mat = np.vstack(
            [np.asarray(data_by_id[cid], dtype=np.float32).reshape(1, -1) for cid in ordered_chunk_ids]
        )
        mat = cls._ensure_float32_2d(mat, dim)

        idx = hnswlib.Index(space=space, dim=dim)
        idx.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        idx.add_items(mat, labels)

        label_to_chunk_id: dict[int, str] = {int(i): cid for i, cid in enumerate(ordered_chunk_ids)}
        chunk_id_to_label: dict[str, int] = {cid: int(i) for i, cid in enumerate(ordered_chunk_ids)}

        bin_path = out_dir / "hnsw.bin"
        idx.save_index(str(bin_path))

        idmap_path = out_dir / "idmap.json"
        idmap_payload = {
            "label_to_chunk_id": {str(k): v for k, v in label_to_chunk_id.items()},
            "chunk_id_to_label": chunk_id_to_label,
        }
        idmap_path.write_text(json.dumps(idmap_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        meta_path = out_dir / "meta.json"
        meta_payload = {
            "pack_id": pack_id,
            "dim": dim,
            "space": space,
            "hnsw_params": {"M": M, "ef_construction": ef_construction},
            "count": n,
            "created_at_utc": cls._utc_now_iso(),
            "schema_version": B3_SCHEMA_VERSION,
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return cls(
            pack_id=pack_id,
            dim=dim,
            space=space,
            index=idx,
            label_to_chunk_id=label_to_chunk_id,
            chunk_id_to_label=chunk_id_to_label,
        )

    @classmethod
    def load(cls, pack_dir: Path) -> "HNSWPackIndex":
        meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
        idmap = json.loads((pack_dir / "idmap.json").read_text(encoding="utf-8"))

        pack_id = cast(str, meta["pack_id"])
        dim = int(meta["dim"])
        space = cast(str, meta.get("space", "cosine"))
        count = int(meta.get("count", 0))

        label_to_chunk_id_raw: dict[str, str] = cast(dict[str, str], idmap["label_to_chunk_id"])
        label_to_chunk_id: dict[int, str] = {int(k): v for k, v in label_to_chunk_id_raw.items()}
        chunk_id_to_label: dict[str, int] = {
            str(k): int(v) for k, v in cast(dict[str, int], idmap["chunk_id_to_label"]).items()
        }

        idx = hnswlib.Index(space=space, dim=dim)
        idx.load_index(str(pack_dir / "hnsw.bin"), max_elements=max(count, 1))

        return cls(
            pack_id=pack_id,
            dim=dim,
            space=space,
            index=idx,
            label_to_chunk_id=label_to_chunk_id,
            chunk_id_to_label=chunk_id_to_label,
        )

    def set_ef(self, ef: int) -> None:
        self.index.set_ef(ef)

    def query(self, vector: "np.ndarray", k: int) -> list[tuple[str, float]]:
        """
        Returns (chunk_id, distance) pairs ordered by best match first.
        For cosine space, distance is the hnswlib cosine distance (lower is better).
        """
        v = self._ensure_float32_2d(vector, self.dim)
        labels, distances = self.index.knn_query(v, k=k)

        out: list[tuple[str, float]] = []
        for lab, dist in zip(labels[0].tolist(), distances[0].tolist(), strict=True):
            cid = self.label_to_chunk_id[int(lab)]
            out.append((cid, float(dist)))
        return out

