"""Generate independently integrated binary128 labels for frozen holdout stencils."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
HORIZON = "50"
ENERGY_LIMIT = 1e-12
TIME_LIMIT_SECONDS = 45
TIERS = {"standard": ("1e-20", "1"), "tight": ("1e-24", "0.5")}
COORDINATE_COLUMNS = ("grid", "i", "j", "x", "y")
OUTPUT_COLUMNS = (*COORDINATE_COLUMNS, "label", "status", "standard_label", "tight_label",
                  "standard_time", "tight_time", "standard_drift", "tight_drift")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def point_key(row):
    return str(row["grid"]), int(row["i"]), int(row["j"])


def read_points(path):
    unique = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(COORDINATE_COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain {', '.join(COORDINATE_COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            grid, i, j = point_key(row)
            x, y = float(row["x"]), float(row["y"])
            if not grid or not (0 <= i < 256 and 0 <= j < 256 and isfinite(x) and isfinite(y)):
                raise ValueError(f"Invalid coordinate at line {line} of {path}.")
            key = (grid, i, j)
            if key in unique:
                previous = unique[key]
                if (float(previous["x"]), float(previous["y"])) != (x, y):
                    raise ValueError(f"Conflicting coordinates at stencil point {key}.")
            else:
                unique[key] = {"grid": grid, "i": i, "j": j, "x": row["x"], "y": row["y"]}
    if not unique:
        raise ValueError("The frozen stencil file contains no points.")
    return unique


def run_native(binary, point, tolerance, cap):
    # strtod in the native oracle receives a hexadecimal spelling of the exact
    # binary64 value that the CUDA solver sees through its decimal CSV parser.
    command = [str(binary), float(point["x"]).hex(), float(point["y"]).hex(),
               HORIZON, tolerance, cap]
    start = time.monotonic()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=TIME_LIMIT_SECONDS)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": time.monotonic() - start}
    if process.returncode not in (0, 1):
        raise RuntimeError(f"Native oracle failed for {point_key(point)}: {process.stderr.strip()}")
    result = json.loads(process.stdout)
    label, stop, drift = result["label"], result["time"], result["energy_drift"]
    if (result["precision_bits"] != 113 or label not in range(6) or
            not 0 <= stop <= float(HORIZON) or not isfinite(drift) or drift < 0):
        raise RuntimeError(f"Invalid binary128 result for {point_key(point)}")
    if (process.returncode == 1) != (label == 5):
        raise RuntimeError(f"Inconsistent exit status for {point_key(point)}")
    return {"status": "ok" if label != 5 else "unresolved",
            "seconds": time.monotonic() - start, "result": result, "stdout": process.stdout.strip()}


def score_point(binary, point):
    tiers = {name: run_native(binary, point, *options) for name, options in TIERS.items()}
    standard, tight = (tiers[name].get("result", {}) for name in TIERS)
    label = 5
    if any(tier["status"] == "timeout" for tier in tiers.values()):
        status = "timeout"
    elif any(tier["status"] != "ok" for tier in tiers.values()):
        status = "integrator_unresolved"
    elif standard["energy_drift"] >= ENERGY_LIMIT or tight["energy_drift"] >= ENERGY_LIMIT:
        status = "energy_drift"
    elif standard["label"] != tight["label"]:
        status = "label_disagreement"
    else:
        status = "resolved"
        label = standard["label"]
    labels = dict(point, label=label, status=status,
                  standard_label=standard.get("label", 5), tight_label=tight.get("label", 5),
                  standard_time=standard.get("time"), tight_time=tight.get("time"),
                  standard_drift=standard.get("energy_drift"), tight_drift=tight.get("energy_drift"))
    return {"row": labels, "tiers": tiers}


def config(args, points):
    return {"cohort_sha256": digest(args.cohort),
            "source_sha256": digest(ROOT / "holdout_oracle.cpp"),
            "runner_sha256": digest(Path(__file__)),
            "points": len(points), "horizon": HORIZON, "cutoff": "0.001",
            "max_relative_energy_drift": ENERGY_LIMIT, "watchdog_seconds_per_solve": TIME_LIMIT_SECONDS,
            "binary64_input_roundtrip": "hexadecimal float -> strtod -> __float128",
            "tiers": {name: {"tolerance": tol, "physical_cap_factor": cap}
                      for name, (tol, cap) in TIERS.items()}}


def prepare(args, locked):
    temporary = args.out / "tmp"
    temporary.mkdir(exist_ok=True)
    binary = args.out / "holdout_oracle"
    argv = ["g++", "-std=gnu++17", "-O3", "-Wall", "-Wextra", str(ROOT / "holdout_oracle.cpp"),
            "-lquadmath", "-o", str(binary)]
    env = dict(os.environ, TMPDIR=str(temporary))
    build = subprocess.run(argv, env=env, check=True, capture_output=True, text=True, timeout=120)
    test = subprocess.run([str(binary), "--self-test"], check=True, capture_output=True, text=True,
                          timeout=TIME_LIMIT_SECONDS)
    fixture = json.loads(test.stdout)
    if not fixture.get("passed") or fixture.get("precision_bits") != 113:
        raise RuntimeError("Equilateral and grazing fixtures must pass before holdout integration.")
    atomic_json(args.out / "manifest.json", {
        **locked, "binary_sha256": digest(binary), "compiler": subprocess.check_output(
            ["g++", "--version"], text=True).splitlines()[0],
        "compile_flags": argv[1:5] + ["-lquadmath"], "compiler_stderr": build.stderr,
        "fixture": fixture})
    print(f"Binary128 equilateral and grazing fixtures passed for {locked['points']} frozen points.",
          flush=True)
    return binary


def read_partial(path, points):
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            row = record["row"]
            key = point_key(row)
            if key not in points or key in completed or any(
                    row[column] != points[key][column] for column in COORDINATE_COLUMNS):
                raise ValueError(f"Unrecognized checkpoint point {key}.")
            completed[key] = row
    return completed


def run(args):
    points = read_points(args.cohort)
    locked = config(args, points)
    manifest_path = args.out / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        binary = args.out / "holdout_oracle"
        if any(previous[key] != value for key, value in locked.items()) or digest(binary) != previous["binary_sha256"]:
            raise ValueError("Cohort, compiled oracle, or locked settings changed since preparation.")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        binary = prepare(args, locked)
    if args.prepare:
        return 0

    raw = args.out / "raw.jsonl"
    completed = read_partial(raw, points)
    pending = [point for key, point in points.items() if key not in completed]
    start = time.monotonic()
    statuses = Counter(row["status"] for row in completed.values())
    with raw.open("a", encoding="utf-8", newline="\n") as records, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(score_point, binary, point) for point in pending]
        for future in as_completed(futures):
            record = future.result()
            row = record["row"]
            records.write(json.dumps(record, allow_nan=False) + "\n")
            records.flush()
            completed[point_key(row)] = row
            statuses[row["status"]] += 1
            if len(completed) % 16 == 0 or len(completed) == len(points):
                print(f"Binary128 {len(completed)}/{len(points)} points, {statuses['resolved']} resolved.",
                      flush=True)

    labels = args.out / "labels.csv"
    staging = labels.with_suffix(".csv.tmp")
    with staging.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(completed[key] for key in points)
    staging.replace(labels)
    atomic_json(args.out / "summary.json", {"points": len(points), "statuses": dict(statuses),
                "new_points_this_invocation": len(pending), "seconds_this_invocation": time.monotonic() - start,
                "labels_sha256": digest(labels), "raw_sha256": digest(raw)})
    return 0


def windows_delegate(args):
    platform = ["wsl.exe", "-d", args.distro, "--exec"]
    def wslpath(path):
        return subprocess.check_output([*platform, "wslpath", "-a", "-u", str(path)], text=True).strip()

    command = [*platform, "python3", "-B", wslpath(Path(__file__).resolve()),
               "--cohort", wslpath(args.cohort), "--out", wslpath(args.out), "--jobs", str(args.jobs)]
    if args.prepare:
        command.append("--prepare")
    if args.resume:
        command.append("--resume")
    return subprocess.call(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path, help="Frozen CSV of unique or repeated stencil points")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--distro", default="Ubuntu-24.04", help="WSL distribution used from Windows")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Compile and verify fixtures without solving holdout points")
    mode.add_argument("--resume", action="store_true", help="Run a prepared directory or resume its checkpoint")
    args = parser.parse_args()
    args.cohort = args.cohort.resolve(strict=True)
    args.out = args.out.resolve()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be in [1,8]")
    if os.name == "nt":
        return windows_delegate(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
