"""SECOM dataset preprocessing.

a single, documented preprocessing entrypoint
that produces *loadable* dataset artifacts.

It can load SECOM directly from a .zip (e.g. secom.zip) without manual extraction.
It searches inside the archive for files ending in:
  - secom.data
  - secom_labels.data

Outputs (in --out directory):
  - secom_processed.npz        : arrays X, y, feature_names, timestamps
  - preprocess_metadata.json   : config, stats, hashes, versions
  - (optional) secom_processed.csv.gz

Example usage:
  python -m group_elephant.workstreams.data.secom_preprocess stats --zip path/to/secom.zip
  python -m group_elephant.workstreams.data.secom_preprocess preprocess --zip path/to/secom.zip --out artifacts/

  # or the classic way with extracted files:
  python -m group_elephant.workstreams.data.secom_preprocess preprocess \
      --data path/to/secom.data --labels path/to/secom_labels.data --out artifacts/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import platform
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

LOGGER = logging.getLogger("secom_preprocess")


# ----------------------------
# Config + artifacts
# ----------------------------

ImputeStrategy = Literal["median", "mean", "most_frequent"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for preprocessing."""

    max_missing_fraction: float = 0.60
    impute_strategy: ImputeStrategy = "median"
    drop_constant_features: bool = True
    scale: bool = True
    dtype: Literal["float32", "float64"] = "float32"
    csv_gz: bool = False


@dataclass(frozen=True)
class DatasetArtifacts:
    """Processed dataset and supporting fields to reload it."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    timestamps: list[str]


# ----------------------------
# Utilities
# ----------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def parse_dtype(dtype: str) -> np.dtype:
    if dtype == "float32":
        return np.float32
    if dtype == "float64":
        return np.float64
    raise ValueError(f"Unsupported dtype: {dtype}")


# ----------------------------
# Loading SECOM (files or zip)
# ----------------------------

def _find_member(zf: zipfile.ZipFile, suffix: str) -> str:
    """Find a single member that endswith suffix; prefer shorter paths if multiple."""
    candidates = [n for n in zf.namelist() if n.lower().endswith(suffix.lower()) and not n.endswith("/")]
    if not candidates:
        raise FileNotFoundError(f"Could not find '{suffix}' inside zip.")
    candidates.sort(key=lambda n: (n.count("/"), len(n)))
    return candidates[0]


def load_secom_from_zip(zip_path: Path) -> tuple[np.ndarray, list[str], np.ndarray, list[str], dict]:
    """Load X, feature_names, y, timestamps from a SECOM .zip archive."""
    ensure_exists(zip_path, "zip file")
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"--zip expects a .zip file, got: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        data_member = _find_member(zf, "secom.data")
        labels_member = _find_member(zf, "secom_labels.data")

        with zf.open(data_member, "r") as f:
            X = np.loadtxt(io.TextIOWrapper(f, encoding="utf-8"), dtype=np.float64)

        y_list: list[int] = []
        ts_list: list[str] = []
        with zf.open(labels_member, "r") as f:
            for line_no, raw in enumerate(io.TextIOWrapper(f, encoding="utf-8"), start=1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid labels line {line_no}: expected 2 columns, got: {parts}")
                y_list.append(int(float(parts[0])))
                ts_list.append(" ".join(parts[1:]))

        y = np.asarray(y_list, dtype=np.int64)

        # Hash actual member bytes for metadata reproducibility
        data_bytes = zf.read(data_member)
        labels_bytes = zf.read(labels_member)
        hashes = {
            "zip_sha256": sha256_file(zip_path),
            "data_member": data_member,
            "labels_member": labels_member,
            "data_member_sha256": sha256_bytes(data_bytes),
            "labels_member_sha256": sha256_bytes(labels_bytes),
        }

    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return X, feature_names, y, ts_list, hashes


def load_secom_features(data_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load SECOM feature matrix from extracted 'secom.data'."""
    ensure_exists(data_path, "data file")
    X = np.loadtxt(str(data_path), dtype=np.float64)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return X, feature_names


def load_secom_labels(labels_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load labels + timestamps from extracted 'secom_labels.data'."""
    ensure_exists(labels_path, "labels file")

    y_list: list[int] = []
    ts_list: list[str] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid labels line {line_no}: expected 2 columns, got: {parts}")
            y_list.append(int(float(parts[0])))
            ts_list.append(" ".join(parts[1:]))

    y = np.asarray(y_list, dtype=np.int64)
    return y, ts_list


def validate_alignment(X: np.ndarray, y: np.ndarray, timestamps: list[str]) -> None:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows.")
    if len(timestamps) != y.shape[0]:
        raise ValueError(f"Row mismatch: timestamps has {len(timestamps)} rows, y has {y.shape[0]} rows.")


# ----------------------------
# Preprocessing steps
# ----------------------------

def drop_high_missing_columns(
    X: np.ndarray, feature_names: list[str], max_missing_fraction: float
) -> tuple[np.ndarray, list[str], np.ndarray]:
    if not (0.0 <= max_missing_fraction <= 1.0):
        raise ValueError("max_missing_fraction must be in [0, 1].")
    missing_frac = np.mean(np.isnan(X), axis=0)
    kept_mask = missing_frac <= max_missing_fraction
    X_new = X[:, kept_mask]
    names_new = [n for n, keep in zip(feature_names, kept_mask) if keep]
    return X_new, names_new, kept_mask


def impute_missing(X: np.ndarray, strategy: ImputeStrategy) -> tuple[np.ndarray, np.ndarray]:
    X_out = X.copy()
    if strategy == "median":
        vals = np.nanmedian(X_out, axis=0)
    elif strategy == "mean":
        vals = np.nanmean(X_out, axis=0)
    elif strategy == "most_frequent":
        vals = np.nanmedian(X_out, axis=0)
    else:
        raise ValueError(f"Unknown impute strategy: {strategy}")

    vals = np.where(np.isnan(vals), 0.0, vals)
    inds = np.where(np.isnan(X_out))
    X_out[inds] = np.take(vals, inds[1])
    return X_out, vals


def drop_constant_features(
    X: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str], np.ndarray]:
    var = np.var(X, axis=0)
    kept_mask = var > 0.0
    X_new = X[:, kept_mask]
    names_new = [n for n, keep in zip(feature_names, kept_mask) if keep]
    return X_new, names_new, kept_mask


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    Xs = (X - mu) / sigma
    return Xs, mu, sigma


# ----------------------------
# Orchestration
# ----------------------------

def _load_inputs(
    data_path: Path | None, labels_path: Path | None, zip_path: Path | None
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], dict]:
    """Load from zip if provided, else from extracted files."""
    if zip_path is not None:
        X, feat_names, y, timestamps, hashes = load_secom_from_zip(zip_path)
        meta_in = {
            "mode": "zip",
            "zip_path": str(zip_path),
            **hashes,
            "n_rows": int(X.shape[0]),
            "n_features_raw": int(X.shape[1]),
        }
        return X, feat_names, y, timestamps, meta_in

    if data_path is None or labels_path is None:
        raise ValueError("Provide either --zip OR both --data and --labels.")

    X, feat_names = load_secom_features(data_path)
    y, timestamps = load_secom_labels(labels_path)
    meta_in = {
        "mode": "files",
        "data_path": str(data_path),
        "labels_path": str(labels_path),
        "data_sha256": sha256_file(data_path),
        "labels_sha256": sha256_file(labels_path),
        "n_rows": int(X.shape[0]),
        "n_features_raw": int(X.shape[1]),
    }
    return X, feat_names, y, timestamps, meta_in


def preprocess_secom(
    data_path: Path | None, labels_path: Path | None, zip_path: Path | None, cfg: PreprocessConfig
) -> tuple[DatasetArtifacts, dict]:
    X_raw, feat_names, y, timestamps, input_meta = _load_inputs(data_path, labels_path, zip_path)
    validate_alignment(X_raw, y, timestamps)

    meta: dict = {
        "input": input_meta,
        "config": asdict(cfg),
        "steps": {},
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }

    X1, names1, kept1 = drop_high_missing_columns(X_raw, feat_names, cfg.max_missing_fraction)
    meta["steps"]["drop_high_missing"] = {
        "max_missing_fraction": cfg.max_missing_fraction,
        "n_features_after": int(X1.shape[1]),
        "dropped": int(np.size(kept1) - int(np.sum(kept1))),
    }

    X2, impute_vals = impute_missing(X1, cfg.impute_strategy)
    meta["steps"]["impute"] = {
        "strategy": cfg.impute_strategy,
        "impute_values_preview": [float(v) for v in impute_vals[: min(10, len(impute_vals))]],
    }

    X3, names3, kept3 = (X2, names1, np.ones(X2.shape[1], dtype=bool))
    if cfg.drop_constant_features:
        X3, names3, kept3 = drop_constant_features(X2, names1)

    meta["steps"]["drop_constant_features"] = {
        "enabled": bool(cfg.drop_constant_features),
        "n_features_after": int(X3.shape[1]),
        "dropped": int(np.size(kept3) - int(np.sum(kept3))),
    }

    X4 = X3
    if cfg.scale:
        X4, mu, sigma = standardize(X3)
        meta["steps"]["standardize"] = {
            "enabled": True,
            "mean_preview": [float(v) for v in mu[: min(10, len(mu))]],
            "std_preview": [float(v) for v in sigma[: min(10, len(sigma))]],
        }
    else:
        meta["steps"]["standardize"] = {"enabled": False}

    out_dtype = parse_dtype(cfg.dtype)
    X_final = X4.astype(out_dtype, copy=False)
    y_final = y.astype(np.int64, copy=False)

    artifacts = DatasetArtifacts(
        X=X_final,
        y=y_final,
        feature_names=names3,
        timestamps=timestamps,
    )

    meta["output"] = {
        "n_rows": int(X_final.shape[0]),
        "n_features": int(X_final.shape[1]),
        "X_dtype": str(X_final.dtype),
        "y_dtype": str(y_final.dtype),
        "label_values": sorted(list({int(v) for v in np.unique(y_final)})),
        "missing_after": float(np.mean(np.isnan(X_final))),
    }
    return artifacts, meta


# ----------------------------
# Saving outputs
# ----------------------------

def save_npz(artifacts: DatasetArtifacts, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "secom_processed.npz"
    np.savez_compressed(
        out_path,
        X=artifacts.X,
        y=artifacts.y,
        feature_names=np.asarray(artifacts.feature_names, dtype=object),
        timestamps=np.asarray(artifacts.timestamps, dtype=object),
    )
    return out_path


def save_metadata(meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocess_metadata.json"
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def save_csv_gz(artifacts: DatasetArtifacts, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "secom_processed.csv.gz"
    header = list(artifacts.feature_names) + ["label", "timestamp"]
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(artifacts.X.shape[0]):
            row = artifacts.X[i, :].tolist()
            row.append(int(artifacts.y[i]))
            row.append(artifacts.timestamps[i])
            writer.writerow(row)
    return out_path


# ----------------------------
# CLI
# ----------------------------

def _add_input_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", type=Path, help="Path to SECOM zip (contains secom.data + secom_labels.data).")
    g.add_argument("--data", type=Path, help="Path to extracted secom.data")
    p.add_argument("--labels", type=Path, help="Path to extracted secom_labels.data (required if --data used)")


def cmd_stats(data_path: Path | None, labels_path: Path | None, zip_path: Path | None) -> int:
    X, feat_names, y, timestamps, input_meta = _load_inputs(data_path, labels_path, zip_path)
    validate_alignment(X, y, timestamps)

    missing_frac = float(np.mean(np.isnan(X)))
    per_col_missing = np.mean(np.isnan(X), axis=0)

    print(
        json.dumps(
            {
                **input_meta,
                "overall_missing_fraction": missing_frac,
                "max_missing_fraction_any_feature": float(np.max(per_col_missing)),
                "min_missing_fraction_any_feature": float(np.min(per_col_missing)),
                "label_values": sorted(list({int(v) for v in np.unique(y)})),
                "feature_name_example": feat_names[:5],
            },
            indent=2,
        )
    )
    return 0


def cmd_preprocess(
    data_path: Path | None, labels_path: Path | None, zip_path: Path | None, out_dir: Path, cfg: PreprocessConfig
) -> int:
    artifacts, meta = preprocess_secom(data_path, labels_path, zip_path, cfg)

    npz_path = save_npz(artifacts, out_dir)
    save_metadata(meta, out_dir)
    if cfg.csv_gz:
        save_csv_gz(artifacts, out_dir)

    print(str(npz_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="secom_preprocess",
        description="Preprocess SECOM into a single loadable artifact (no train/test split). Zip supported.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="Print basic stats about SECOM input.")
    _add_input_args(p_stats)

    p_run = sub.add_parser("preprocess", help="Preprocess and write artifacts.")
    _add_input_args(p_run)
    p_run.add_argument("--out", required=True, type=Path, help="Output directory for artifacts")

    p_run.add_argument("--max-missing-fraction", type=float, default=0.60)
    p_run.add_argument("--impute-strategy", choices=["median", "mean", "most_frequent"], default="median")
    p_run.add_argument("--drop-constant", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p_run.add_argument("--csv-gz", action=argparse.BooleanOptionalAction, default=False)

    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging(args.verbose)

    if args.cmd == "stats":
        if args.data is not None and args.labels is None:
            raise SystemExit("--labels is required when using --data.")
        return cmd_stats(args.data, args.labels, args.zip)

    if args.cmd == "preprocess":
        if args.data is not None and args.labels is None:
            raise SystemExit("--labels is required when using --data.")
        cfg = PreprocessConfig(
            max_missing_fraction=float(args.max_missing_fraction),
            impute_strategy=args.impute_strategy,
            drop_constant_features=bool(args.drop_constant),
            scale=bool(args.scale),
            dtype=args.dtype,
            csv_gz=bool(args.csv_gz),
        )
        return cmd_preprocess(args.data, args.labels, args.zip, args.out, cfg)

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
