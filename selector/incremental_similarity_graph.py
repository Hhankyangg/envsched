"""Sparse similarity access and persistent incremental top-k graph state.

This module keeps the online graph path free of dense ``N x N`` similarity
matrices.  Static TF-IDF views are persisted as CSR matrices, while each graph
plan stores only ``K + 1`` candidates per target.  The extra candidate detects
ties at the selection boundary; tied targets are resolved with the legacy
``numpy.argpartition`` rule over their complete active candidate row.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

PROVIDER_SCHEMA_VERSION = 1

GRAPH_STATE_SCHEMA_VERSION = 1

ABSTRACT_TOPOLOGY_SCHEMA_VERSION = 1

GRAPH_SCOPES = ("all", "within", "cross")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_save_sparse(path: Path, matrix: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    with temporary.open("wb") as handle:
        sparse.save_npz(handle, matrix, compressed=True)
    os.replace(temporary, path)


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_similarity_text(value: Any) -> str:
    """Replace identifiers, dates, and numbers with placeholder tokens."""
    text = str(value).lower()
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,36}\b", " uuid_token ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " date_token ", text)
    text = re.sub(
        r"\b(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9][a-z0-9_-]{2,}\b",
        " id_token ",
        text,
    )
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " number_token ", text)
    return re.sub(r"\s+", " ", text).strip()


def abstract_topology_text(topology: dict[str, Any]) -> str:
    """Keep execution/reward shape while dropping domain and schema identity."""
    tokens: list[str] = []
    for operation in topology.get("operations", []):
        tokens.extend(
            [
                f"operation_{operation.get('family')}",
                "operation_dependency_bin_"
                f"{min(len(operation.get('depends_on_operations', [])), 4)}",
                f"operation_gate_bin_{min(len(operation.get('gate_ids', [])), 5)}",
            ]
        )
    for gate in topology.get("gates", []):
        tokens.extend(
            [
                f"gate_{gate.get('type')}",
                f"provenance_{gate.get('provenance')}",
                f"reachability_{gate.get('reachability')}",
                f"observability_{gate.get('observability')}",
                f"recoverability_{gate.get('recoverability')}",
            ]
        )
    tokens.extend(
        f"reward_{group.get('relation')}"
        for group in topology.get("reward_groups", [])
    )
    tokens.extend(
        f"conflict_{conflict.get('kind')}"
        for conflict in topology.get("conflicts", [])
    )
    tokens.extend(
        [
            f"operation_count_{min(len(topology.get('operations', [])), 20)}",
            f"gate_count_{min(len(topology.get('gates', [])), 25)}",
            f"reward_count_{min(len(topology.get('reward_groups', [])), 15)}",
            f"branch_count_{min(len(topology.get('independent_branches', [])), 10)}",
        ]
    )
    return " ".join(tokens)


@dataclass(frozen=True)
class TfidfViewSpec:
    name: str
    weight: float
    texts: Sequence[str]
    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 35_000

    def normalized_texts(self) -> list[str]:
        return [normalize_similarity_text(text) for text in self.texts]


@dataclass(frozen=True)
class _SparseView:
    name: str
    weight: float
    matrix: sparse.csr_matrix


@runtime_checkable
class SimilarityProvider(Protocol):
    size: int
    cache_key: str

    def columns(self, column_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        ...

    def submatrix(
        self,
        row_indices: Sequence[int] | np.ndarray,
        column_indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        ...


class SparseTfidfSimilarityProvider:
    """Exact float32 view blend backed by normalized sparse TF-IDF matrices."""

    def __init__(
        self,
        views: Sequence[_SparseView],
        *,
        cache_key: str,
        cache_dir: str | Path | None = None,
        force_unit_diagonal: bool = True,
    ) -> None:
        if not views:
            raise ValueError("At least one TF-IDF view is required")
        sizes = {int(view.matrix.shape[0]) for view in views}
        if len(sizes) != 1:
            raise ValueError("All TF-IDF views must have the same row count")
        if any(view.matrix.dtype != np.float64 for view in views):
            raise ValueError("Exact legacy compatibility requires float64 CSR data")
        if any(not math.isfinite(view.weight) or view.weight < 0 for view in views):
            raise ValueError("TF-IDF view weights must be finite and non-negative")
        # cosine_similarity re-normalizes even TF-IDF's already normalized rows.
        # Do that exact second normalization once, then reuse it for every MMR
        # column and neighbor insertion. Cached matrices stay in their original
        # form, so reloading does not accumulate normalization roundoff.
        self._views = tuple(
            _SparseView(view.name, view.weight, normalize(view.matrix, copy=True))
            for view in views
        )
        self.size = sizes.pop()
        self.shape = (self.size, self.size)
        self.cache_key = str(cache_key)
        self.cache_dir = None if cache_dir is None else str(Path(cache_dir))
        self.force_unit_diagonal = bool(force_unit_diagonal)
        self.query_count = 0
        self.queried_column_count = 0

    def __len__(self) -> int:
        return self.size

    @property
    def view_weights(self) -> dict[str, float]:
        return {view.name: float(view.weight) for view in self._views}

    @property
    def feature_nnz(self) -> int:
        return sum(int(view.matrix.nnz) for view in self._views)

    @property
    def feature_bytes(self) -> int:
        return sum(
            int(
                view.matrix.data.nbytes
                + view.matrix.indices.nbytes
                + view.matrix.indptr.nbytes
            )
            for view in self._views
        )

    def _indices(
        self,
        indices: Sequence[int] | np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        result = np.asarray(indices, dtype=np.int64)
        if result.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if len(result) and (result.min() < 0 or result.max() >= self.size):
            raise IndexError(f"{name} contains an out-of-range task index")
        return result

    def columns(self, column_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        columns = self._indices(column_indices, name="column_indices")
        self.query_count += 1
        self.queried_column_count += len(columns)
        result = np.zeros((self.size, len(columns)), dtype=np.float32)
        if not len(columns):
            return result
        for view in self._views:
            contribution = np.clip(
                safe_sparse_dot(view.matrix, view.matrix[columns].T, dense_output=True),
                0.0,
                1.0,
            ).astype(np.float32)
            result += view.weight * contribution
        result = np.clip(result, 0.0, 1.0)
        if self.force_unit_diagonal:
            result[columns, np.arange(len(columns))] = 1.0
        return result

    def submatrix(
        self,
        row_indices: Sequence[int] | np.ndarray,
        column_indices: Sequence[int] | np.ndarray,
    ) -> np.ndarray:
        rows = self._indices(row_indices, name="row_indices")
        columns = self._indices(column_indices, name="column_indices")
        self.query_count += 1
        self.queried_column_count += len(columns)
        result = np.zeros((len(rows), len(columns)), dtype=np.float32)
        if not len(rows) or not len(columns):
            return result
        for view in self._views:
            contribution = np.clip(
                safe_sparse_dot(view.matrix[rows], view.matrix[columns].T, dense_output=True),
                0.0,
                1.0,
            ).astype(np.float32)
            result += view.weight * contribution
        result = np.clip(result, 0.0, 1.0)
        if self.force_unit_diagonal:
            row_positions = {int(value): position for position, value in enumerate(rows)}
            for column_position, task_index in enumerate(columns):
                row_position = row_positions.get(int(task_index))
                if row_position is not None:
                    result[row_position, column_position] = 1.0
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "provider": "sparse_tfidf",
            "cache_key": self.cache_key,
            "cache_dir": self.cache_dir,
            "task_count": self.size,
            "view_count": len(self._views),
            "view_weights": self.view_weights,
            "feature_nnz": self.feature_nnz,
            "feature_bytes": self.feature_bytes,
            "query_count": self.query_count,
            "queried_column_count": self.queried_column_count,
            "dense_similarity_materialized": False,
        }

SimilaritySource = np.ndarray | SimilarityProvider


def _source_size(source: SimilaritySource) -> int:
    if isinstance(source, np.ndarray):
        if source.ndim != 2 or source.shape[0] != source.shape[1]:
            raise ValueError("Dense similarity source must be a square matrix")
        return int(source.shape[0])
    return int(source.size)


def similarity_columns(
    source: SimilaritySource,
    column_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    columns = np.asarray(column_indices, dtype=np.int64)
    if columns.ndim != 1:
        raise ValueError("column_indices must be one-dimensional")
    if isinstance(source, np.ndarray):
        return source[:, columns]
    return source.columns(columns)


def similarity_column(source: SimilaritySource, column_index: int) -> np.ndarray:
    return similarity_columns(source, np.asarray([column_index], dtype=np.int64))[:, 0]


def similarity_submatrix(
    source: SimilaritySource,
    row_indices: Sequence[int] | np.ndarray,
    column_indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    columns = np.asarray(column_indices, dtype=np.int64)
    if rows.ndim != 1 or columns.ndim != 1:
        raise ValueError("row_indices and column_indices must be one-dimensional")
    if isinstance(source, np.ndarray):
        return source[np.ix_(rows, columns)]
    return source.submatrix(rows, columns)


def similarity_row(
    source: SimilaritySource,
    row_index: int,
    column_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    if column_indices is None:
        column_indices = np.arange(_source_size(source), dtype=np.int64)
    return similarity_submatrix(
        source,
        np.asarray([row_index], dtype=np.int64),
        column_indices,
    )[0]


def _provider_cache_key(
    specs: Sequence[TfidfViewSpec],
    *,
    force_unit_diagonal: bool,
) -> str:
    digest = hashlib.sha256(f"sparse-tfidf-v{PROVIDER_SCHEMA_VERSION}".encode())
    digest.update(
        json.dumps(
            {
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "sklearn_version": sklearn.__version__,
                "force_unit_diagonal": bool(force_unit_diagonal),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for spec in specs:
        normalized = spec.normalized_texts()
        payload = {
            "name": spec.name,
            "weight": spec.weight,
            "analyzer": spec.analyzer,
            "ngram_range": spec.ngram_range,
            "max_features": spec.max_features,
            "text_hash": _hash_strings(normalized),
            "text_count": len(normalized),
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def build_sparse_similarity_provider(
    view_specs: Sequence[TfidfViewSpec],
    cache_dir: str | Path,
    *,
    namespace: str,
    force_unit_diagonal: bool = True,
) -> SparseTfidfSimilarityProvider:
    """Build or load an exact sparse TF-IDF provider."""
    specs = tuple(view_specs)
    if not specs:
        raise ValueError("At least one view spec is required")
    task_counts = {len(spec.texts) for spec in specs}
    if len(task_counts) != 1:
        raise ValueError("All view specs must contain one text per task")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("TF-IDF view names must be unique")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", namespace):
        raise ValueError("namespace contains unsupported characters")

    root = Path(cache_dir)
    metadata_path = root / f"{namespace}.metadata.json"
    cache_key = _provider_cache_key(
        specs,
        force_unit_diagonal=force_unit_diagonal,
    )
    metadata: dict[str, Any] | None = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            metadata = None

    expected_names = [spec.name for spec in specs]
    can_load = bool(
        metadata
        and metadata.get("schema_version") == PROVIDER_SCHEMA_VERSION
        and metadata.get("cache_key") == cache_key
        and metadata.get("task_count") == task_counts.copy().pop()
        and [item.get("name") for item in metadata.get("views", [])]
        == expected_names
    )
    views: list[_SparseView] = []
    if can_load:
        try:
            for spec, item in zip(specs, metadata["views"]):
                matrix = sparse.load_npz(root / str(item["file"])).tocsr()
                if matrix.shape[0] != len(spec.texts) or matrix.dtype != np.float64:
                    raise ValueError("Cached TF-IDF matrix has an incompatible shape")
                views.append(_SparseView(spec.name, float(spec.weight), matrix))
        except (
            OSError,
            EOFError,
            ValueError,
            KeyError,
            zipfile.BadZipFile,
        ):
            views = []

    if not views:
        root.mkdir(parents=True, exist_ok=True)
        view_metadata = []
        for position, spec in enumerate(specs):
            matrix = TfidfVectorizer(
                lowercase=True,
                analyzer=spec.analyzer,
                ngram_range=spec.ngram_range,
                min_df=1,
                max_features=spec.max_features,
                sublinear_tf=True,
            ).fit_transform(spec.normalized_texts()).tocsr()
            if matrix.dtype != np.float64:
                matrix = matrix.astype(np.float64)
            filename = f"{namespace}.view_{position:02d}.{spec.name}.npz"
            _atomic_save_sparse(root / filename, matrix)
            views.append(_SparseView(spec.name, float(spec.weight), matrix))
            view_metadata.append(
                {
                    "name": spec.name,
                    "weight": float(spec.weight),
                    "file": filename,
                    "shape": list(matrix.shape),
                    "nnz": int(matrix.nnz),
                    "data_dtype": str(matrix.dtype),
                }
            )
        _atomic_write_json(
            metadata_path,
            {
                "schema_version": PROVIDER_SCHEMA_VERSION,
                "cache_key": cache_key,
                "task_count": task_counts.pop(),
                "force_unit_diagonal": bool(force_unit_diagonal),
                "library_versions": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                },
                "views": view_metadata,
            },
        )
    return SparseTfidfSimilarityProvider(
        views,
        cache_key=cache_key,
        cache_dir=root,
        force_unit_diagonal=force_unit_diagonal,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _optional_static_texts(
    path: str,
    task_ids: list[str],
    field_name: str | None,
) -> list[str] | None:
    artifact = Path(path)
    if not artifact.exists():
        return None
    records = _read_jsonl(artifact)
    by_id = {str(item.get("task_id")): item for item in records}
    if set(by_id) != set(task_ids):
        return None
    values = []
    for task_id in task_ids:
        record = by_id[task_id]
        value = record.get(field_name, record) if field_name else record
        values.append(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return values


def _input_request_key(
    namespace: str,
    task_ids: Sequence[str],
    paths: Sequence[str],
    scenarios: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Fingerprint raw inputs before expensive text extraction/normalization.

    Hash file contents (not only mtimes), ordered task IDs, dependency versions,
    and feature implementation sources. This shortcut is only for loading an
    already validated feature cache; its provider key remains the full text key.
    """
    digest = hashlib.sha256(namespace.encode())
    digest.update(_hash_strings(task_ids).encode())
    digest.update(json.dumps([PROVIDER_SCHEMA_VERSION, np.__version__,
                              scipy.__version__, sklearn.__version__]).encode())
    if scenarios is not None:
        digest.update(json.dumps(scenarios, ensure_ascii=True, sort_keys=True).encode())
    source_paths = [str(Path(__file__).resolve())]
    if scenarios is not None:
        import task_features
        source_paths.append(str(Path(task_features.__file__).resolve()))
    for name in [*paths, *source_paths]:
        path = Path(name)
        digest.update(str(path.resolve()).encode())
        if path.exists():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def _load_provider_for_request(
    cache_dir: str | Path, namespace: str, request_key: str, task_count: int,
) -> SparseTfidfSimilarityProvider | None:
    root = Path(cache_dir)
    try:
        metadata = json.loads((root / f"{namespace}.metadata.json").read_text())
        if (metadata.get("input_request_key") != request_key
                or metadata.get("schema_version") != PROVIDER_SCHEMA_VERSION
                or metadata.get("task_count") != task_count):
            return None
        views = []
        for item in metadata["views"]:
            matrix = sparse.load_npz(root / item["file"]).tocsr()
            if (list(matrix.shape) != item["shape"] or matrix.shape[0] != task_count
                    or matrix.dtype != np.float64 or matrix.nnz != item["nnz"]):
                return None
            views.append(_SparseView(item["name"], float(item["weight"]), matrix))
        return SparseTfidfSimilarityProvider(
            views, cache_key=metadata["cache_key"], cache_dir=root,
        )
    except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return None


def _save_request_key(cache_dir: str | Path, namespace: str, request_key: str) -> None:
    path = Path(cache_dir) / f"{namespace}.metadata.json"
    metadata = json.loads(path.read_text())
    metadata["input_request_key"] = request_key
    _atomic_write_json(path, metadata)


def build_static_similarity_provider(
    pool: Any,
    scenarios: Sequence[dict[str, Any]],
    topology_path: str,
    cache_dir: str | Path,
) -> tuple[SparseTfidfSimilarityProvider, dict[str, float]]:
    """Build the dense selector's static multiview similarity as sparse features."""
    from task_features import state_signature, verifier_text

    task_ids = pool["task_id"].astype(str).tolist()
    namespace = "static_similarity_features"
    request_key = _input_request_key(
        namespace, task_ids, [topology_path], scenarios,
    )
    cached = _load_provider_for_request(cache_dir, namespace, request_key, len(task_ids))
    if cached is not None:
        return cached, cached.view_weights

    task_texts: list[str] = []
    verifier_natural: list[str] = []
    verifier_structural: list[str] = []
    static_context: list[str] = []
    for scenario in scenarios:
        task = str(scenario["task"])
        natural, structural, _, _ = verifier_text(scenario)
        state, _, _ = state_signature(scenario.get("init_config", {}), task)
        task_texts.append(task)
        verifier_natural.append(natural)
        verifier_structural.append(structural)
        static_context.append(f"TASK {task} CHECKS {natural} STATE {state}")

    raw: list[tuple[str, float, Sequence[str], str, tuple[int, int]]] = [
        ("task_word", 0.35, task_texts, "word", (1, 2)),
        ("task_char", 0.10, task_texts, "char_wb", (3, 5)),
        ("verifier_natural", 0.20, verifier_natural, "word", (1, 2)),
        ("verifier_structural", 0.15, verifier_structural, "word", (1, 2)),
        ("static_context", 0.20, static_context, "word", (1, 2)),
    ]
    topology = _optional_static_texts(topology_path, task_ids, "topology")
    if topology is not None:
        raw.append(("llm_topology", 0.25, topology, "word", (1, 2)))
    total_weight = sum(item[1] for item in raw)
    weights = {name: weight / total_weight for name, weight, *_ in raw}
    specs = [
        TfidfViewSpec(
            name=name,
            weight=weights[name],
            texts=texts,
            analyzer=analyzer,
            ngram_range=ngram_range,
        )
        for name, _, texts, analyzer, ngram_range in raw
    ]
    provider = build_sparse_similarity_provider(
        specs,
        cache_dir,
        namespace=namespace,
    )
    _save_request_key(cache_dir, namespace, request_key)
    return provider, weights


def build_abstract_topology_similarity_provider(
    pool: Any,
    topology_path: str,
    cache_dir: str | Path,
) -> SparseTfidfSimilarityProvider:
    """Build the exact abstract cross-environment view as sparse features."""
    task_ids = pool["task_id"].astype(str).tolist()
    namespace = f"abstract_topology_similarity_features_v{ABSTRACT_TOPOLOGY_SCHEMA_VERSION}"
    request_key = _input_request_key(namespace, task_ids, [topology_path])
    cached = _load_provider_for_request(cache_dir, namespace, request_key, len(task_ids))
    if cached is not None:
        return cached
    records = _read_jsonl(Path(topology_path))
    by_id = {str(item.get("task_id")): item for item in records}
    if set(by_id) != set(task_ids):
        raise ValueError(
            "Abstract topology graph requires exactly one topology for every task"
        )
    texts = [
        abstract_topology_text(by_id[task_id].get("topology", by_id[task_id]))
        for task_id in task_ids
    ]
    provider = build_sparse_similarity_provider(
        [TfidfViewSpec("abstract_topology", 1.0, texts)],
        cache_dir,
        namespace=namespace,
    )
    _save_request_key(cache_dir, namespace, request_key)
    return provider


@dataclass
class _NeighborPlan:
    name: str
    task_count: int
    k: int
    scope: str
    positive_only: bool
    provider_cache_key: str
    environment_hash: str
    active_mask: np.ndarray
    candidate_indices: np.ndarray
    candidate_scores: np.ndarray
    selected_indices: np.ndarray
    selected_scores: np.ndarray
    boundary_tie_mask: np.ndarray
    history_max_step: int | None = None
    last_audit: dict[str, Any] = field(default_factory=dict)


class IncrementalGraphState:
    """Persistent K+1 top-k plans for one ordered task pool."""

    def __init__(
        self,
        state_dir: str | Path,
        task_ids: Sequence[str],
        *,
        rebuild_on_error: bool = True,
        target_chunk_size: int = 2048,
        candidate_chunk_size: int = 64,
    ) -> None:
        if not isinstance(target_chunk_size, int) or target_chunk_size <= 0:
            raise ValueError("target_chunk_size must be a positive integer")
        if not isinstance(candidate_chunk_size, int) or candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be a positive integer")
        self.state_dir = Path(state_dir)
        self.task_ids = tuple(str(task_id) for task_id in task_ids)
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task_ids must be unique")
        self.task_count = len(self.task_ids)
        self.task_ids_hash = _hash_strings(self.task_ids)
        self.rebuild_on_error = bool(rebuild_on_error)
        self.target_chunk_size = target_chunk_size
        self.candidate_chunk_size = candidate_chunk_size
        self._plans: dict[str, _NeighborPlan] = {}
        self._load_attempted: set[str] = set()
        self._load_errors: dict[str, str] = {}
        self._known_plan_names: set[str] = set()
        self._prepared_request_keys: dict[str, tuple[Any, ...]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        path = self.state_dir / "metadata.json"
        if path.exists():
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if metadata.get("schema_version") != GRAPH_STATE_SCHEMA_VERSION:
                    raise ValueError("graph metadata schema mismatch")
                if metadata.get("task_ids_hash") != self.task_ids_hash:
                    raise ValueError("graph metadata ordered task pool mismatch")
                if int(metadata.get("task_count", -1)) != self.task_count:
                    raise ValueError("graph metadata task count mismatch")
                self._known_plan_names.update(
                    str(name) for name in metadata.get("plan_names", [])
                )
                self._load_errors.update(
                    {
                        str(name): str(error)
                        for name, error in metadata.get("load_errors", {}).items()
                    }
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                if not self.rebuild_on_error:
                    raise
                self._load_errors["metadata"] = f"{type(error).__name__}: {error}"
        self._write_metadata()

    def _write_metadata(self) -> None:
        _atomic_write_json(
            self.state_dir / "metadata.json",
            {
                "schema_version": GRAPH_STATE_SCHEMA_VERSION,
                "task_count": self.task_count,
                "task_ids_hash": self.task_ids_hash,
                "target_chunk_size": self.target_chunk_size,
                "candidate_chunk_size": self.candidate_chunk_size,
                "plan_names": sorted(self._known_plan_names | set(self._plans)),
                "load_errors": self._load_errors,
            },
        )

    @staticmethod
    def _validate_plan_name(plan_name: str) -> str:
        name = str(plan_name)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("plan_name contains unsupported characters")
        return name

    def _plan_path(self, plan_name: str) -> Path:
        return self.state_dir / f"plan.{self._validate_plan_name(plan_name)}.npz"

    def _empty_plan(
        self,
        *,
        plan_name: str,
        provider: SimilaritySource,
        k: int,
        scope: str,
        positive_only: bool,
        env_ids: np.ndarray,
    ) -> _NeighborPlan:
        width = k + 1
        provider_key = getattr(provider, "cache_key", "dense_array")
        return _NeighborPlan(
            name=plan_name,
            task_count=self.task_count,
            k=k,
            scope=scope,
            positive_only=positive_only,
            provider_cache_key=str(provider_key),
            environment_hash=_hash_strings(env_ids.astype(str).tolist()),
            active_mask=np.zeros(self.task_count, dtype=bool),
            candidate_indices=np.full(
                (self.task_count, width), -1, dtype=np.int32
            ),
            candidate_scores=np.full(
                (self.task_count, width), -np.inf, dtype=np.float32
            ),
            selected_indices=np.full(
                (self.task_count, k), -1, dtype=np.int32
            ),
            selected_scores=np.full(
                (self.task_count, k), -np.inf, dtype=np.float32
            ),
            boundary_tie_mask=np.zeros(self.task_count, dtype=bool),
        )

    def _load_plan(self, plan_name: str) -> _NeighborPlan | None:
        if plan_name in self._load_attempted:
            return self._plans.get(plan_name)
        self._load_attempted.add(plan_name)
        path = self._plan_path(plan_name)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                config = json.loads(str(payload["config"].item()))
                if config["schema_version"] != GRAPH_STATE_SCHEMA_VERSION:
                    raise ValueError("graph state schema mismatch")
                if config["task_ids_hash"] != self.task_ids_hash:
                    raise ValueError("ordered task pool mismatch")
                if int(config.get("task_count", -1)) != self.task_count:
                    raise ValueError("graph state task count mismatch")
                if str(config.get("name")) != plan_name:
                    raise ValueError("graph state plan name mismatch")
                k = int(config["k"])
                if k <= 0:
                    raise ValueError("stored graph k must be positive")
                if str(config.get("scope")) not in GRAPH_SCOPES:
                    raise ValueError("stored graph scope is invalid")
                plan = _NeighborPlan(
                    name=plan_name,
                    task_count=self.task_count,
                    k=k,
                    scope=str(config["scope"]),
                    positive_only=bool(config["positive_only"]),
                    provider_cache_key=str(config["provider_cache_key"]),
                    environment_hash=str(config["environment_hash"]),
                    active_mask=payload["active_mask"].astype(bool, copy=False),
                    candidate_indices=payload["candidate_indices"].astype(
                        np.int32, copy=False
                    ),
                    candidate_scores=payload["candidate_scores"].astype(
                        np.float32, copy=False
                    ),
                    selected_indices=payload["selected_indices"].astype(
                        np.int32, copy=False
                    ),
                    selected_scores=payload["selected_scores"].astype(
                        np.float32, copy=False
                    ),
                    boundary_tie_mask=payload["boundary_tie_mask"].astype(
                        bool, copy=False
                    ),
                    history_max_step=(
                        None
                        if config.get("history_max_step") is None
                        else int(config["history_max_step"])
                    ),
                    last_audit=dict(config.get("last_audit", {})),
                )
            self._validate_loaded_shapes(plan)
            self._plans[plan_name] = plan
            self._known_plan_names.add(plan_name)
            return plan
        except (
            OSError,
            EOFError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ) as error:
            if not self.rebuild_on_error:
                raise
            self._load_errors[plan_name] = f"{type(error).__name__}: {error}"
            self._write_metadata()
            return None

    def _validate_loaded_shapes(self, plan: _NeighborPlan) -> None:
        expected = {
            "active_mask": (self.task_count,),
            "candidate_indices": (self.task_count, plan.k + 1),
            "candidate_scores": (self.task_count, plan.k + 1),
            "selected_indices": (self.task_count, plan.k),
            "selected_scores": (self.task_count, plan.k),
            "boundary_tie_mask": (self.task_count,),
        }
        for name, shape in expected.items():
            if getattr(plan, name).shape != shape:
                raise ValueError(
                    f"Stored {name} shape mismatch: "
                    f"{getattr(plan, name).shape} != {shape}"
                )
        for name in ("candidate_indices", "selected_indices"):
            indices = getattr(plan, name)
            if np.any(indices < -1) or np.any(indices >= self.task_count):
                raise ValueError(f"Stored {name} contains out-of-range indices")
            referenced = indices[indices >= 0]
            if len(referenced) and not np.all(plan.active_mask[referenced]):
                raise ValueError(f"Stored {name} references inactive candidates")
        if np.any(np.isnan(plan.candidate_scores)) or np.any(
            np.isnan(plan.selected_scores)
        ):
            raise ValueError("Stored graph scores contain NaN")

    def _save_plan(self, plan: _NeighborPlan) -> None:
        self._known_plan_names.add(plan.name)
        config = {
            "schema_version": GRAPH_STATE_SCHEMA_VERSION,
            "task_ids_hash": self.task_ids_hash,
            "task_count": self.task_count,
            "name": plan.name,
            "k": plan.k,
            "scope": plan.scope,
            "positive_only": plan.positive_only,
            "provider_cache_key": plan.provider_cache_key,
            "environment_hash": plan.environment_hash,
            "history_max_step": plan.history_max_step,
            "last_audit": plan.last_audit,
        }
        _atomic_save_npz(
            self._plan_path(plan.name),
            config=np.asarray(json.dumps(config, sort_keys=True)),
            active_mask=plan.active_mask,
            candidate_indices=plan.candidate_indices,
            candidate_scores=plan.candidate_scores,
            selected_indices=plan.selected_indices,
            selected_scores=plan.selected_scores,
            boundary_tie_mask=plan.boundary_tie_mask,
        )

    def save(self) -> None:
        for plan in self._plans.values():
            self._save_plan(plan)
        self._write_metadata()

    def _validate_request(
        self,
        provider: SimilaritySource,
        active_mask: np.ndarray,
        k: int,
        scope: str,
        env_ids: Sequence[str] | np.ndarray,
        plan_name: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        name = self._validate_plan_name(plan_name)
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if scope not in GRAPH_SCOPES:
            raise ValueError(f"scope must be one of {GRAPH_SCOPES}")
        if _source_size(provider) != self.task_count:
            raise ValueError("Similarity source does not match the ordered task pool")
        active = np.asarray(active_mask, dtype=bool)
        environments = np.asarray(env_ids).astype(str)
        if active.shape != (self.task_count,):
            raise ValueError("active_mask must have shape (task_count,)")
        if environments.shape != (self.task_count,):
            raise ValueError("env_ids must have shape (task_count,)")
        return active, environments, name

    @staticmethod
    def _plan_matches(
        plan: _NeighborPlan,
        provider: SimilaritySource,
        k: int,
        scope: str,
        positive_only: bool,
        environments: np.ndarray,
    ) -> bool:
        return bool(
            plan.k == k
            and plan.scope == scope
            and plan.positive_only == positive_only
            and plan.provider_cache_key
            == str(getattr(provider, "cache_key", "dense_array"))
            and plan.environment_hash
            == _hash_strings(environments.astype(str).tolist())
        )

    def _merge_candidate_block(
        self,
        plan: _NeighborPlan,
        candidate_ids: np.ndarray,
        columns: np.ndarray,
        environments: np.ndarray,
    ) -> None:
        width = plan.k + 1
        for start in range(0, self.task_count, self.target_chunk_size):
            stop = min(start + self.target_chunk_size, self.task_count)
            targets = np.arange(start, stop, dtype=np.int32)
            block_scores = columns[start:stop].astype(np.float32, copy=True)
            eligible = targets[:, None] != candidate_ids[None, :]
            if plan.scope == "within":
                eligible &= (
                    environments[targets, None]
                    == environments[candidate_ids][None, :]
                )
            elif plan.scope == "cross":
                eligible &= (
                    environments[targets, None]
                    != environments[candidate_ids][None, :]
                )
            if plan.positive_only:
                eligible &= block_scores > 0.0
            block_scores[~eligible] = -np.inf
            new_indices = np.broadcast_to(
                candidate_ids.astype(np.int32), block_scores.shape
            ).copy()
            new_indices[~eligible] = -1
            combined_indices = np.concatenate(
                [plan.candidate_indices[start:stop], new_indices], axis=1
            )
            combined_scores = np.concatenate(
                [plan.candidate_scores[start:stop], block_scores], axis=1
            )
            order = np.lexsort((combined_indices, -combined_scores), axis=1)
            keep = order[:, :width]
            plan.candidate_indices[start:stop] = np.take_along_axis(
                combined_indices, keep, axis=1
            )
            plan.candidate_scores[start:stop] = np.take_along_axis(
                combined_scores, keep, axis=1
            )

    def _add_candidates(
        self,
        plan: _NeighborPlan,
        provider: SimilaritySource,
        candidate_ids: np.ndarray,
        environments: np.ndarray,
    ) -> None:
        for start in range(0, len(candidate_ids), self.candidate_chunk_size):
            block = candidate_ids[start : start + self.candidate_chunk_size]
            columns = similarity_columns(provider, block)
            self._merge_candidate_block(plan, block, columns, environments)
            plan.active_mask[block] = True

    def _eligible_candidates(
        self,
        plan: _NeighborPlan,
        target: int,
        environments: np.ndarray,
        provider: SimilaritySource,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.flatnonzero(plan.active_mask)
        candidates = candidates[candidates != target]
        if plan.scope == "within":
            candidates = candidates[environments[candidates] == environments[target]]
        elif plan.scope == "cross":
            candidates = candidates[environments[candidates] != environments[target]]
        if not len(candidates):
            return candidates, np.asarray([], dtype=np.float32)
        scores = similarity_row(provider, target, candidates).astype(
            np.float32, copy=False
        )
        if plan.positive_only:
            positive = scores > 0.0
            candidates = candidates[positive]
            scores = scores[positive]
        return candidates, scores

    def _refresh_selected(
        self,
        plan: _NeighborPlan,
        provider: SimilaritySource,
        environments: np.ndarray,
    ) -> int:
        plan.selected_indices.fill(-1)
        plan.selected_scores.fill(-np.inf)
        valid = plan.candidate_indices >= 0
        counts = valid.sum(axis=1)
        take = min(plan.k, plan.candidate_indices.shape[1])
        plan.selected_indices[:, :take] = plan.candidate_indices[:, :take]
        plan.selected_scores[:, :take] = plan.candidate_scores[:, :take]
        for target, count in enumerate(counts):
            if count < plan.k:
                plan.selected_indices[target, int(count) :] = -1
                plan.selected_scores[target, int(count) :] = -np.inf
        plan.boundary_tie_mask[:] = (
            (counts > plan.k)
            & (plan.candidate_scores[:, plan.k - 1] == plan.candidate_scores[:, plan.k])
        )
        tie_targets = np.flatnonzero(plan.boundary_tie_mask)
        for target in tie_targets:
            candidates, scores = self._eligible_candidates(
                plan, int(target), environments, provider
            )
            count = min(plan.k, len(candidates))
            if not count:
                continue
            positions = np.argpartition(-scores, count - 1)[:count]
            plan.selected_indices[target, :count] = candidates[positions]
            plan.selected_scores[target, :count] = scores[positions]
            if count < plan.k:
                plan.selected_indices[target, count:] = -1
                plan.selected_scores[target, count:] = -np.inf
        return int(len(tie_targets))

    def prepare_plan(
        self,
        provider: SimilaritySource,
        active_mask: Sequence[bool] | np.ndarray,
        k: int,
        scope: str,
        env_ids: Sequence[str] | np.ndarray,
        plan_name: str,
        *,
        positive_only: bool = False,
        history_max_step: int | None = None,
    ) -> dict[str, Any]:
        active, environments, name = self._validate_request(
            provider, np.asarray(active_mask), k, scope, env_ids, plan_name
        )
        if history_max_step is not None and history_max_step < 0:
            raise ValueError("history_max_step must be non-negative or None")
        plan = self._plans.get(name) or self._load_plan(name)
        rebuild_reason: str | None = None
        if plan is None:
            rebuild_reason = self._load_errors.get(name, "state_missing")
        elif not self._plan_matches(
            plan, provider, k, scope, positive_only, environments
        ):
            rebuild_reason = "plan_configuration_changed"
        elif np.any(plan.active_mask & ~active):
            rebuild_reason = "active_mask_shrank"
        elif (
            history_max_step is not None
            and plan.history_max_step is not None
            and plan.history_max_step > history_max_step
        ):
            rebuild_reason = "state_ahead_of_history"

        if rebuild_reason is not None:
            plan = self._empty_plan(
                plan_name=name,
                provider=provider,
                k=k,
                scope=scope,
                positive_only=positive_only,
                env_ids=environments,
            )
            self._plans[name] = plan
            action = "rebuild"
            new_candidates = np.flatnonzero(active)
        else:
            new_candidates = np.flatnonzero(active & ~plan.active_mask)
            action = "incremental_update" if len(new_candidates) else "reuse"

        query_count_before = int(getattr(provider, "query_count", 0))
        queried_columns_before = int(getattr(provider, "queried_column_count", 0))
        if len(new_candidates):
            self._add_candidates(plan, provider, new_candidates, environments)
        tie_count = self._refresh_selected(plan, provider, environments)
        plan.active_mask = active.copy()
        plan.history_max_step = history_max_step
        audit = {
            "plan_name": name,
            "action": action,
            "rebuild_reason": rebuild_reason,
            "task_count": self.task_count,
            "active_task_count": int(active.sum()),
            "new_active_task_count": int(len(new_candidates)),
            "k": k,
            "stored_candidate_count_per_target": k + 1,
            "scope": scope,
            "positive_only": bool(positive_only),
            "boundary_tie_fallback_target_count": tie_count,
            "history_max_step": history_max_step,
            "target_chunk_size": self.target_chunk_size,
            "candidate_chunk_size": self.candidate_chunk_size,
            "provider_query_count_delta": (
                int(getattr(provider, "query_count", 0)) - query_count_before
            ),
            "provider_queried_column_count_delta": (
                int(getattr(provider, "queried_column_count", 0))
                - queried_columns_before
            ),
            "dense_similarity_materialized": isinstance(provider, np.ndarray),
        }
        plan.last_audit = audit
        self._save_plan(plan)
        self._write_metadata()
        self._prepared_request_keys[name] = (
            id(provider),
            id(active_mask),
            k,
            scope,
            id(env_ids),
            bool(positive_only),
            history_max_step,
        )
        return dict(audit)

    def neighbors(
        self,
        target: int,
        provider: SimilaritySource,
        active_mask: Sequence[bool] | np.ndarray,
        k: int,
        scope: str,
        env_ids: Sequence[str] | np.ndarray,
        plan_name: str,
        *,
        positive_only: bool = False,
        history_max_step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        name = self._validate_plan_name(plan_name)
        if target < 0 or target >= self.task_count:
            raise IndexError("target is outside the ordered task pool")
        request_key = (
            id(provider),
            id(active_mask),
            k,
            scope,
            id(env_ids),
            bool(positive_only),
            history_max_step,
        )
        if self._prepared_request_keys.get(name) != request_key:
            self.prepare_plan(
                provider,
                active_mask,
                k,
                scope,
                env_ids,
                name,
                positive_only=positive_only,
                history_max_step=history_max_step,
            )
        return self.prepared_neighbors(target, name)

    def prepared_neighbors(
        self,
        target: int,
        plan_name: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Read one target from an explicitly prepared plan in ``O(K)`` time."""
        name = self._validate_plan_name(plan_name)
        if target < 0 or target >= self.task_count:
            raise IndexError("target is outside the ordered task pool")
        plan = self._plans.get(name)
        if plan is None:
            raise RuntimeError(
                f"Graph plan {name!r} is not prepared; call prepare_plan first"
            )
        valid = plan.selected_indices[target] >= 0
        return (
            plan.selected_indices[target, valid].astype(int, copy=True),
            plan.selected_scores[target, valid].astype(np.float32, copy=True),
            dict(plan.last_audit),
        )

    def audit(self) -> dict[str, Any]:
        plan_audits = {
            name: dict(plan.last_audit) for name, plan in sorted(self._plans.items())
        }
        state_bytes = 0
        for plan in self._plans.values():
            state_bytes += sum(
                int(array.nbytes)
                for array in (
                    plan.active_mask,
                    plan.candidate_indices,
                    plan.candidate_scores,
                    plan.selected_indices,
                    plan.selected_scores,
                    plan.boundary_tie_mask,
                )
            )
        return {
            "schema_version": GRAPH_STATE_SCHEMA_VERSION,
            "state_dir": str(self.state_dir),
            "task_count": self.task_count,
            "task_ids_hash": self.task_ids_hash,
            "target_chunk_size": self.target_chunk_size,
            "candidate_chunk_size": self.candidate_chunk_size,
            "plan_count": len(self._plans),
            "state_array_bytes": state_bytes,
            "load_errors": dict(self._load_errors),
            "plans": plan_audits,
        }
