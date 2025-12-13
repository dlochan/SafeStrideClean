import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


def _is_within_dir(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def extract_zip_posix(zip_path: Path, out_root: Path, force: bool) -> None:
    if not zip_path.exists():
        raise SystemExit(f"Zip not found: {zip_path}")

    out_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            raw_name = info.filename
            if not raw_name:
                continue

            norm_name = raw_name.replace("\\", "/")
            p = PurePosixPath(norm_name)

            if p.is_absolute() or ".." in p.parts:
                raise SystemExit(f"Refusing to extract unsafe path: {raw_name} -> {norm_name}")

            dest = out_root.joinpath(*p.parts)
            if not _is_within_dir(dest, out_root):
                raise SystemExit(f"Refusing to extract outside output root: {dest}")

            is_dir = norm_name.endswith("/") or getattr(info, "is_dir", lambda: False)()
            if is_dir:
                dest.mkdir(parents=True, exist_ok=True)
                continue

            if dest.exists() and not force:
                raise SystemExit(f"Refusing to overwrite existing file (use --force): {dest}")

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

            perm = (info.external_attr >> 16) & 0o777
            if perm:
                try:
                    os.chmod(dest, perm)
                except OSError:
                    pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract a Windows-created Vast bundle zip on Linux by normalizing \\\\ to / in zip member names"
    )
    ap.add_argument("--zip", dest="zip_path", default="vast_bundle_fz_bw.zip", help="Path to vast_bundle_fz_bw.zip")
    ap.add_argument(
        "--out-root",
        dest="out_root",
        default="/workspace",
        help="Extraction root directory (e.g. /workspace)",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    out_root = Path(args.out_root)

    extract_zip_posix(zip_path=zip_path, out_root=out_root, force=bool(args.force))

    expected = out_root / "vast_bundle_fz_bw" / "data_root" / "manifests"
    if not expected.is_dir():
        raise SystemExit(f"ERROR: expected manifests dir not found: {expected}")

    cfg_path = out_root / "vast_bundle_fz_bw" / "vnext_example.vast.yaml"
    print("OK: extracted bundle")
    print(f"manifests_dir={expected}")
    print(f"bundled_config={cfg_path}")
    print("Correct paths.data_root to use on Vast:")
    print("/workspace/vast_bundle_fz_bw/data_root")


if __name__ == "__main__":
    sys.exit(main())
