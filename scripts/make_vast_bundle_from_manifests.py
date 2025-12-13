from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


def _read_yaml(path: Path) -> Dict:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"Config is not a mapping: {path}")
    return obj


def _as_posix_rel(p: Path) -> str:
    return p.as_posix()


def _resolve_manifest_path(manifest_value: str, data_root_abs: Path, repo_root: Path) -> Path:
    raw = str(manifest_value).strip().replace("\\", "/")
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    cand_repo = repo_root / p
    if cand_repo.exists():
        return cand_repo
    cand_data = data_root_abs / p
    if cand_data.exists():
        return cand_data
    return cand_repo


def _resolve_data_file_path(path_value: str, data_root_abs: Path, repo_root: Path) -> Path:
    raw = str(path_value).strip().replace("\\", "/")
    p = Path(raw)
    if p.is_absolute():
        return p
    cand_repo = repo_root / p
    if cand_repo.exists():
        return cand_repo
    cand_data = data_root_abs / p
    if cand_data.exists():
        return cand_data
    return cand_repo


def _dest_rel_for_group(
    src_abs: Path,
    data_root_abs: Path,
    group: str,
) -> Path:
    src_abs = src_abs.resolve()
    data_root_abs = data_root_abs.resolve()
    rel: Optional[Path] = None
    try:
        rel = src_abs.relative_to(data_root_abs)
    except Exception:
        rel = Path(src_abs.name)

    if rel.parts and rel.parts[0].lower() == group.lower():
        rel = Path(*rel.parts[1:])
    return Path(group) / rel


def _iter_manifest_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {path}")
        for row in reader:
            yield {k: ("" if v is None else str(v)) for k, v in row.items()}


def _write_manifest_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _build_bundle(
    base_config: Path,
    out_dir: Path,
    overwrite: bool,
) -> Tuple[Path, Path, Path, int, int]:
    repo_root = Path.cwd().resolve()
    cfg = _read_yaml(base_config)

    paths_cfg = cfg.get("paths", {}) or {}
    data_root = paths_cfg.get("data_root")
    if not data_root:
        raise SystemExit("configs.paths.data_root is missing")

    data_root_abs = (repo_root / Path(str(data_root))).resolve() if not Path(str(data_root)).is_absolute() else Path(str(data_root)).resolve()

    data_cfg = cfg.get("data", {}) or {}
    train_manifest_val = data_cfg.get("train_manifest")
    val_manifest_val = data_cfg.get("val_manifest")

    if not train_manifest_val and not val_manifest_val:
        manifest_val = data_cfg.get("manifest")
        if not manifest_val:
            raise SystemExit("Config.data must provide train_manifest/val_manifest or manifest")
        train_manifest_val = manifest_val

    train_manifest_abs = _resolve_manifest_path(str(train_manifest_val), data_root_abs, repo_root)
    val_manifest_abs = (
        _resolve_manifest_path(str(val_manifest_val), data_root_abs, repo_root)
        if val_manifest_val
        else None
    )

    if not train_manifest_abs.exists():
        raise SystemExit(f"Train manifest not found: {train_manifest_abs}")
    if val_manifest_abs is not None and not val_manifest_abs.exists():
        raise SystemExit(f"Val manifest not found: {val_manifest_abs}")

    if out_dir.exists():
        if not overwrite:
            raise SystemExit(f"out_dir already exists: {out_dir} (use --overwrite to replace)")
        shutil.rmtree(out_dir)

    bundle_root = out_dir
    bundle_data_root = bundle_root / "data_root"
    imu_dir = bundle_data_root / "imu"
    grf_dir = bundle_data_root / "grf"
    manifests_dir = bundle_data_root / "manifests"
    meta_dir = bundle_data_root / "meta"
    for p in (imu_dir, grf_dir, manifests_dir, meta_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Snapshot config
    snapshot_path = meta_dir / "source_config_snapshot.yaml"
    _copy_file(base_config, snapshot_path)

    used_manifests: List[Path] = [train_manifest_abs]
    if val_manifest_abs is not None:
        used_manifests.append(val_manifest_abs)

    # Copy original manifest files
    for m in used_manifests:
        _copy_file(m, manifests_dir / m.name)

    # Copy referenced data files and build rewrite mapping
    src_to_bundle_rel: Dict[str, str] = {}
    copied_imu = 0
    copied_grf = 0

    bundle_prefix = Path(out_dir.name) / "data_root"

    def ensure_copy(src_abs: Path, group: str, trial_id: str) -> str:
        nonlocal copied_imu, copied_grf
        src_key = str(src_abs.resolve()).lower()
        if src_key in src_to_bundle_rel:
            return src_to_bundle_rel[src_key]

        dest_rel = _dest_rel_for_group(src_abs, data_root_abs, group=group)
        dest_abs = bundle_data_root / dest_rel

        if dest_abs.exists():
            alt = Path(group) / f"{trial_id}__{src_abs.name}"
            dest_rel = alt
            dest_abs = bundle_data_root / dest_rel

        if not src_abs.exists():
            raise SystemExit(f"Missing referenced file: {src_abs}")
        _copy_file(src_abs, dest_abs)

        if group.lower() == "imu":
            copied_imu += 1
        else:
            copied_grf += 1

        rel_posix = _as_posix_rel(dest_rel)
        src_to_bundle_rel[src_key] = rel_posix
        return rel_posix

    def process_manifest(src_manifest: Path, out_name: str) -> None:
        # Read header to preserve ordering
        with src_manifest.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise SystemExit(f"Manifest has no header: {src_manifest}")
            fieldnames = list(reader.fieldnames)
            rows_out: List[Dict[str, str]] = []
            for row in reader:
                r = {k: ("" if v is None else str(v)) for k, v in row.items()}
                trial_id = r.get("trial_id", "")
                imu_path = r.get("imu_path", "")
                grf_path = r.get("grf_path", "")

                if not imu_path:
                    raise SystemExit(f"Row missing imu_path in {src_manifest} trial_id={trial_id}")

                imu_src = _resolve_data_file_path(imu_path, data_root_abs, repo_root)
                imu_rel = ensure_copy(imu_src, group="imu", trial_id=trial_id or "trial")
                r["imu_path"] = _as_posix_rel(bundle_prefix / Path(imu_rel))

                if grf_path.strip():
                    grf_src = _resolve_data_file_path(grf_path, data_root_abs, repo_root)
                    grf_rel = ensure_copy(grf_src, group="grf", trial_id=trial_id or "trial")
                    r["grf_path"] = _as_posix_rel(bundle_prefix / Path(grf_rel))

                rows_out.append(r)

        out_path = manifests_dir / out_name
        _write_manifest_rows(out_path, fieldnames=fieldnames, rows=rows_out)

    process_manifest(train_manifest_abs, "train_manifest.bundled.csv")
    if val_manifest_abs is not None:
        process_manifest(val_manifest_abs, "val_manifest.bundled.csv")

    # README
    readme_path = bundle_root / "README.txt"
    dt = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    used_manifest_lines = "\n".join([f"- {p}" for p in used_manifests])
    readme = (
        f"Date: {dt}\n"
        f"\n"
        f"Original manifests used:\n{used_manifest_lines}\n"
        f"\n"
        f"Copied IMU files: {copied_imu}\n"
        f"Copied GRF files: {copied_grf}\n"
        f"\n"
        f"Recommended Vast paths.data_root: /workspace/vast_bundle_fz_bw/data_root\n"
        f"NOTE: Bundled manifest imu_path/grf_path are written as paths relative to /workspace (repo root), "
        f"e.g. vast_bundle_fz_bw/data_root/imu/... because the current dataset loader uses the manifest paths as-is.\n"
        f"Bundled config path on Vast: /workspace/vast_bundle_fz_bw/vnext_example.vast.yaml\n"
        f"Bundled manifests:\n"
        f"- manifests/train_manifest.bundled.csv\n"
        f"- manifests/val_manifest.bundled.csv\n"
    )
    readme_path.write_text(readme, encoding="utf-8")

    # Small meta bundle for debugging
    meta_info = {
        "created": dt,
        "source_config": str(base_config),
        "source_data_root": str(data_root_abs),
        "source_manifests": [str(p) for p in used_manifests],
        "bundled_data_root": str(bundle_data_root),
        "bundled_manifests": {
            "train": "manifests/train_manifest.bundled.csv",
            "val": "manifests/val_manifest.bundled.csv" if val_manifest_abs is not None else None,
        },
        "counts": {"imu": copied_imu, "grf": copied_grf},
    }
    (meta_dir / "bundle_meta.json").write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    vast_cfg = _read_yaml(base_config)
    vast_cfg.setdefault("paths", {})
    vast_cfg.setdefault("data", {})
    vast_cfg["paths"]["data_root"] = str(bundle_data_root.as_posix())
    vast_cfg["data"]["train_manifest"] = "manifests/train_manifest.bundled.csv"
    if val_manifest_abs is not None:
        vast_cfg["data"]["val_manifest"] = "manifests/val_manifest.bundled.csv"
    vast_cfg_path = bundle_root / f"{base_config.stem}.vast.yaml"
    vast_cfg_path.write_text(yaml.safe_dump(vast_cfg, sort_keys=False), encoding="utf-8")

    return train_manifest_abs, (val_manifest_abs or train_manifest_abs), bundle_data_root, copied_imu, copied_grf


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a minimal Vast bundle from manifests referenced by a config")
    ap.add_argument("--base-config", required=True, help="Base config (e.g. configs/vnext_example.yaml)")
    ap.add_argument("--out-dir", required=True, help="Output bundle directory (e.g. vast_bundle_fz_bw)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite out-dir if it exists")
    args = ap.parse_args()

    base_config = Path(args.base_config)
    if not base_config.exists():
        raise SystemExit(f"base-config not found: {base_config}")

    out_dir = Path(args.out_dir)

    train_m, val_m, bundle_data_root, n_imu, n_grf = _build_bundle(
        base_config=base_config,
        out_dir=out_dir,
        overwrite=bool(args.overwrite),
    )

    print("OK: Vast bundle created")
    print(f"out_dir={out_dir.resolve()}")
    print(f"bundle_data_root={bundle_data_root.resolve()}")
    print(f"source_train_manifest={train_m}")
    print(f"source_val_manifest={val_m if val_m != train_m else ''}")
    print(f"copied_imu={n_imu} copied_grf={n_grf}")
    print("Use in config:")
    print("paths.data_root: /workspace/vast_bundle_fz_bw/data_root")
    print("data.train_manifest: manifests/train_manifest.bundled.csv")
    print("data.val_manifest: manifests/val_manifest.bundled.csv")


if __name__ == "__main__":
    main()
