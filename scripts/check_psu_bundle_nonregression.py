#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_FILES: List[str] = [
    "bundle_manifest.txt",
    "imu_to_grf_batch_output.json",
    "imu_to_grf_batch_perf.txt",
    "imu_to_grf_output.json",
    "imu_to_grf_perf.txt",
    "provenance.txt",
]

SINGLE_SCHEMA_VERSION = "imu_grf_v1"
SINGLE_OUTPUT_SHAPE = [64, 256, 1]
SINGLE_STATS_KEYS = ["finite_fraction", "min", "max", "mean", "std"]
SINGLE_INPUT_CHANNELS = 12
SINGLE_MODEL = "vnext_fz"

BATCH_SCHEMA_VERSION = "imu_grf_batch_v1"
BATCH_NUM_FILES = 2
BATCH_NUM_OK = 2
BATCH_NUM_FAILED = 0
BATCH_FIRST_OUTPUT_SHAPE = [64, 256, 1]
BATCH_RESULT_REQUIRED_KEYS = ["imu_csv", "ok", "model", "output", "perf"]
BATCH_PERF_REQUIRED_KEYS = [
    "total_ms",
    "forward_ms",
    "rss_mb",
    "build_input_ms",
]

PROVENANCE_SUBSTRINGS: List[str] = [
    "BATCH:",
    "num_files_expected=2",
    "batch_schema_version_expected=imu_grf_batch_v1",
]


def build_expected_spec() -> Dict[str, Any]:
    """Return the frozen PSU bundle contract spec used for non-regression.

    This spec intentionally contains only stable, semantic expectations and no
    dynamic values such as timestamps, git hashes, or absolute paths.
    """

    return {
        "required_files": list(REQUIRED_FILES),
        "single": {
            "schema_version": SINGLE_SCHEMA_VERSION,
            "output_shape": list(SINGLE_OUTPUT_SHAPE),
            "stats_keys": list(SINGLE_STATS_KEYS),
            "input_channels": int(SINGLE_INPUT_CHANNELS),
            "model": SINGLE_MODEL,
        },
        "batch": {
            "schema_version": BATCH_SCHEMA_VERSION,
            "num_files": int(BATCH_NUM_FILES),
            "num_ok": int(BATCH_NUM_OK),
            "num_failed": int(BATCH_NUM_FAILED),
            "first_output_shape": list(BATCH_FIRST_OUTPUT_SHAPE),
            "result_required_keys": list(BATCH_RESULT_REQUIRED_KEYS),
            "perf_required_keys": list(BATCH_PERF_REQUIRED_KEYS),
        },
        "provenance": {
            "substrings": list(PROVENANCE_SUBSTRINGS),
        },
    }


def die(reason: str) -> None:
    print(f"FAIL psu_bundle_contract: {reason}")
    raise SystemExit(1)


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        die(f"missing JSON file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        die(f"invalid JSON in {path}: {exc}")


def _ensure_int_list(value: Any, *, context: str) -> List[int]:
    try:
        seq = list(value)
    except TypeError:
        die(f"{context} is not a sequence")
    out: List[int] = []
    for v in seq:
        try:
            out.append(int(v))
        except Exception:
            die(f"{context} contains non-integer value: {v!r}")
    return out


def check_manifest(out_dir: Path, spec: Dict[str, Any]) -> None:
    path = out_dir / "bundle_manifest.txt"
    if not path.is_file():
        die(f"missing bundle_manifest.txt in {out_dir}")

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # The manifest lists the bundle payload files, not itself. Enforce that the
    # entries exactly match the required files minus "bundle_manifest.txt",
    # sorted lexicographically.
    required_files = list(spec["required_files"])
    expected_manifest_entries = sorted(
        f for f in required_files if f != "bundle_manifest.txt"
    )

    if lines != expected_manifest_entries:
        die(
            "bundle_manifest.txt entries "
            f"{lines!r} != expected {expected_manifest_entries!r}"
        )

    if lines != sorted(lines):
        die("bundle_manifest.txt is not sorted")


def check_single_json(out_dir: Path, spec: Dict[str, Any]) -> None:
    path = out_dir / "imu_to_grf_output.json"
    data = _load_json(path)

    s_spec = spec["single"]

    if data.get("schema_version") != s_spec["schema_version"]:
        die(
            f"imu_to_grf_output.json schema_version={data.get('schema_version')!r} "
            f"!= expected {s_spec['schema_version']!r}"
        )

    output = data.get("output")
    if not isinstance(output, dict):
        die("imu_to_grf_output.json missing output block")

    shape = _ensure_int_list(output.get("shape"), context="single output.shape")
    expected_shape = list(s_spec["output_shape"])
    if shape != expected_shape:
        die(f"single output.shape={shape!r} != expected {expected_shape!r}")

    stats = output.get("stats")
    if not isinstance(stats, dict):
        die("single output.stats missing or not an object")

    missing_stats = [k for k in s_spec["stats_keys"] if k not in stats]
    if missing_stats:
        die(f"single output.stats missing keys {missing_stats!r}")

    input_block = data.get("input")
    if not isinstance(input_block, dict):
        die("imu_to_grf_output.json missing input block")

    channels_raw = input_block.get("channels")
    try:
        channels = int(channels_raw)
    except Exception:
        die(f"input.channels not an int: {channels_raw!r}")

    if channels != int(s_spec["input_channels"]):
        die(
            f"input.channels={channels!r} != expected "
            f"{int(s_spec['input_channels'])!r}"
        )

    model = data.get("model")
    if model != s_spec["model"]:
        die(f"model={model!r} != expected {s_spec['model']!r}")


def check_batch_json(out_dir: Path, spec: Dict[str, Any]) -> None:
    path = out_dir / "imu_to_grf_batch_output.json"
    data = _load_json(path)

    b_spec = spec["batch"]

    if data.get("schema_version") != b_spec["schema_version"]:
        die(
            f"imu_to_grf_batch_output.json schema_version={data.get('schema_version')!r} "
            f"!= expected {b_spec['schema_version']!r}"
        )

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        die("batch metadata missing or not an object")

    def get_meta_int(key: str) -> int:
        raw = metadata.get(key)
        try:
            return int(raw)
        except Exception:
            die(f"metadata.{key} not an int: {raw!r}")

    num_files = get_meta_int("num_files")
    num_ok = get_meta_int("num_ok")
    num_failed = get_meta_int("num_failed")

    if num_files != int(b_spec["num_files"]):
        die(f"metadata.num_files={num_files} != expected {b_spec['num_files']}")
    if num_ok != int(b_spec["num_ok"]):
        die(f"metadata.num_ok={num_ok} != expected {b_spec['num_ok']}")
    if num_failed != int(b_spec["num_failed"]):
        die(f"metadata.num_failed={num_failed} != expected {b_spec['num_failed']}")

    results = data.get("results")
    if not isinstance(results, list) or not results:
        die("batch results missing or empty")

    if len(results) != num_files:
        die(f"len(results)={len(results)} != metadata.num_files={num_files}")

    first = results[0]
    if not isinstance(first, dict):
        die("batch results[0] is not an object")

    required_keys = list(b_spec["result_required_keys"])
    missing_first = [k for k in required_keys if k not in first]
    if missing_first:
        die(f"batch results[0] missing keys {missing_first!r}")

    first_output = first.get("output")
    if not isinstance(first_output, dict):
        die("batch results[0].output missing or not an object")

    first_shape = _ensure_int_list(
        first_output.get("shape"), context="batch first output.shape"
    )
    expected_first_shape = list(b_spec["first_output_shape"])
    if first_shape != expected_first_shape:
        die(
            f"batch first output.shape={first_shape!r} != expected "
            f"{expected_first_shape!r}"
        )

    perf_required_keys = list(b_spec["perf_required_keys"])

    for idx, res in enumerate(results):
        if not isinstance(res, dict):
            die(f"batch result at index {idx} is not an object")

        missing = [k for k in required_keys if k not in res]
        if missing:
            die(f"batch result {idx} missing keys {missing!r}")

        perf = res.get("perf")
        if not isinstance(perf, dict):
            die(f"batch result {idx} missing perf block")

        missing_perf = [k for k in perf_required_keys if k not in perf]
        if missing_perf:
            die(f"batch result {idx} perf missing keys {missing_perf!r}")


def check_provenance(out_dir: Path, spec: Dict[str, Any]) -> None:
    path = out_dir / "provenance.txt"
    if not path.is_file():
        die(f"missing provenance.txt in {out_dir}")

    text = path.read_text(encoding="utf-8")
    for substring in spec["provenance"]["substrings"]:
        if substring not in text:
            die(f"provenance.txt missing substring {substring!r}")


def verify_out_dir(out_dir: Path, spec: Dict[str, Any]) -> None:
    check_manifest(out_dir, spec)
    check_single_json(out_dir, spec)
    check_batch_json(out_dir, spec)
    check_provenance(out_dir, spec)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check PSU bundle contract for non-regression, or write the "
            "baseline JSON for the current contract."
        )
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Path to PSU bundle OUT_DIR produced by psu_bundle_and_verify.sh",
    )
    parser.add_argument(
        "--baseline",
        help="Path to existing PSU bundle contract baseline JSON to enforce.",
    )
    parser.add_argument(
        "--write-baseline",
        help="Write PSU bundle contract baseline JSON to this path and exit.",
    )

    args = parser.parse_args(argv)

    has_baseline = args.baseline is not None
    has_write = args.write_baseline is not None
    if has_baseline == has_write:
        # Either both provided or neither provided.
        parser.error("must specify exactly one of --baseline or --write-baseline")

    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        die(f"OUT_DIR does not exist or is not a directory: {out_dir}")

    spec = build_expected_spec()

    if args.write_baseline is not None:
        # Validate the current OUT_DIR against the contract before freezing it.
        verify_out_dir(out_dir, spec)

        baseline_path = Path(args.write_baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_json = json.dumps(spec, indent=2, sort_keys=True)
        baseline_path.write_text(baseline_json + "\n", encoding="utf-8")
        print(f"Wrote PSU bundle contract baseline to {baseline_path}")
        return 0

    # Enforce that the on-disk baseline matches the hard-coded spec so code and
    # data cannot silently diverge.
    baseline_path = Path(args.baseline)
    try:
        baseline_text = baseline_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        die(f"baseline JSON not found: {baseline_path}")

    try:
        baseline_data = json.loads(baseline_text)
    except json.JSONDecodeError as exc:
        die(f"invalid baseline JSON in {baseline_path}: {exc}")

    expected = build_expected_spec()
    if baseline_data != expected:
        die(
            "baseline JSON does not match expected PSU bundle contract; "
            "regenerate with --write-baseline"
        )

    verify_out_dir(out_dir, spec)

    print(f"PSU_BUNDLE_CONTRACT baseline: {baseline_path}")
    print(f"PSU_BUNDLE_CONTRACT current: {out_dir}")
    print("PASS psu_bundle_contract")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())
