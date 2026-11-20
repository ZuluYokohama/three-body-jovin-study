"""Build an allowlisted public study bundle from finished local runs."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile


SOURCE = (
    ".gitignore", ".jovin/project.json", ".jovin/run_cpu.ps1",
    "boundary_refine.py", "build_cuda.ps1", "cuda/filter.cu",
    "cuda/levi_civita.hpp", "cuda/solver.cu", "ext_precision.cpp",
    "ext_precision.ps1", "holdout.py", "holdout_gpu.py", "holdout_oracle.cpp",
    "holdout_oracle.py", "verify_holdout.ps1", "lc_checks.py", "precision_audit.py",
    "public_release.py", "reference.py", "refined_study.py", "study.py",
    "qa/CMakeLists.txt", "qa/native_ast.json", "qa/native_plan.json",
    "qa/native_trace.json", "qa/verify.cpp",
    "docs/public-readme.md", "docs/public-rights.md", "docs/technical-note.md",
    "docs/reproduce.md",
)
PILOTS = (
    ("study", "pilot-edge", ("cpu_audit.csv", "interior_probes.csv", "outcome_grids.npz",
                             "outcome_map.svg", "manifest.json", "summary.json", "report.md")),
    ("study-lc", "pilot-lc", ("cpu_audit.csv", "paired_audit.csv", "interior_probes.csv",
                              "outcome_grids.npz", "outcome_map.svg", "manifest.json",
                              "summary.json", "report.md")),
    ("study-lc-refined", "pilot-refined", ("dense_precision.csv", "outcome_grids.npz",
                                           "outcome_map.svg", "manifest.json",
                                           "summary.json", "report.md")),
    ("precision-audit-refined", "pilot-precision", ("audit.csv", "gpu_predictions.csv",
                                                    "manifest.json", "summary.json")),
)
HOLDOUT = (
    "protocol.json", "report.md", "summary.json", "scored_centers.csv", "scoring_centers.csv",
    "cohort.csv", "timings.csv", "strict_gpu_predictions.csv", "oracle/labels.csv",
    "oracle/manifest.json", "grid_1_map.svg", "grid_2_map.svg",
    "grid_3_map.svg", *(f"grid_{index}/{name}" for index in range(1, 4)
                          for name in ("timing_summary.json", "outcome_grids.npz")),
)
RECEIPT_FIELDS = (
    "project_id", "status", "passed", "acceptance_status", "file_count",
    "source_manifest_sha256", "receipt_id", "receipt_sha256", "ast_receipt_id",
    "native_receipt_id", "trace_receipt_id",
)
GATE_FIELDS = ("id", "status", "passed", "integrity", "cpp_status", "language_status")
CHAIN_FIELDS = ("valid", "count", "head_sha256")
DEVICE_ID = re.compile(r"(?i)\b(?:GPU|MIG)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b")
LOCAL_PATH = re.compile(
    r"(?i)(?<![\w])(?:[a-z]:[\\/]|\\\\(?:wsl\.localhost|wsl\$)[\\/])"
    r"|(?<![\w/])/(?:mnt/[a-z]|home|Users|private/var|var/tmp)/"
)
SECRETS = re.compile(
    r"(?:\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|"
    r"\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|"
    r"(?i:authorization\s*:\s*bearer\s+)[A-Za-z0-9._-]{12,})"
)
PRIVATE_KEYS = {
    "device_serial", "device_uuid", "gpu_uuid", "host", "hostname", "machine_id",
    "machine_name", "machine_uuid", "serial", "user", "username", "uuid",
}
MD_LINK = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def filtered_scalar(value):
    if LOCAL_PATH.search(value):
        return "[local path omitted]"
    return DEVICE_ID.sub("[device identifier omitted]", value)


def filtered_json(value):
    if isinstance(value, dict):
        result = {}
        local_key_number = 0
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in PRIVATE_KEYS:
                continue
            if LOCAL_PATH.search(key):
                local_key_number += 1
                key = f"[local path {local_key_number} omitted]"
            result[key] = filtered_json(item)
        return result
    if isinstance(value, list):
        return [filtered_json(item) for item in value]
    if isinstance(value, str):
        return filtered_scalar(value)
    return value


def ensure_public_text(value, label):
    # A Windows install directory is a portable build fallback.
    portable_defaults = value.replace("C:/Program Files", "[Windows programs]")
    if LOCAL_PATH.search(portable_defaults):
        raise ValueError(f"Local path remains in {label}")
    if DEVICE_ID.search(value):
        raise ValueError(f"Device identifier remains in {label}")
    if SECRETS.search(value):
        raise ValueError(f"Potential credential in {label}")


def filtered_markdown(text, destination, included):
    def strip_omitted_link(match):
        label, target = match.groups()
        link = target.split("#", 1)[0].split("?", 1)[0]
        if (not link or link.startswith(("https://", "http://")) or
                (destination.parent / link).resolve() in included):
            return match.group()
        return label

    lines = []
    for line in text.splitlines():
        if LOCAL_PATH.search(line):
            lines.append("[Line containing a local path omitted from the public view.]")
        else:
            lines.append(MD_LINK.sub(strip_omitted_link, filtered_scalar(line)))
    return "\n".join(lines) + "\n"


def public_svg(source, raw):
    """Accept static figures, including the PNG data embedded by the outcome renderer."""
    svg = ET.fromstring(raw)
    for element in svg.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag in {"script", "foreignobject", "object", "iframe", "style"}:
            raise ValueError(f"Active content appears in {source.name}: {tag}")
        for attribute, value in element.attrib.items():
            attribute = attribute.rsplit("}", 1)[-1].lower()
            if attribute.startswith("on"):
                raise ValueError(f"Event handler appears in {source.name}")
            if attribute in {"href", "src"} and not value.startswith(("#", "data:image/png;base64,")):
                raise ValueError(f"External resource appears in {source.name}")
            if "url(" in value.lower() and not re.fullmatch(r"url\(#[A-Za-z0-9_-]+\)", value):
                raise ValueError(f"External CSS resource appears in {source.name}")
    return raw


def public_npz(source):
    """Verify the archive contains named NumPy arrays, with no pickle payloads."""
    import numpy as np

    with ZipFile(source) as container:
        entries = container.infolist()
        if (len(entries) > 32 or sum(entry.file_size for entry in entries) > 8_000_000 or
                any(not re.fullmatch(r"[A-Za-z0-9_]+\.npy", entry.filename) for entry in entries)):
            raise ValueError(f"Unexpected array layout in {source.name}")
    with np.load(source, allow_pickle=False) as arrays:
        if any(arrays[key].dtype.kind not in "biuf" for key in arrays.files):
            raise ValueError(f"Non-numeric array in {source.name}")
    return source.read_bytes()


def transfer(source, destination, included, *, original=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if original:
        blob = source.read_bytes()
        ensure_public_text(blob.decode("utf-8-sig"), destination.name)
        destination.write_bytes(blob)
    elif suffix == ".json":
        write_json(destination, filtered_json(read_json(source)))
        ensure_public_text(destination.read_text(encoding="utf-8"), destination.name)
    elif suffix == ".md":
        text = filtered_markdown(source.read_text(encoding="utf-8-sig"), destination, included)
        ensure_public_text(text, destination.name)
        destination.write_text(text, encoding="utf-8")
    elif suffix == ".csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        with source.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.reader(stream):
                writer.writerow([filtered_scalar(cell) for cell in row])
        public = buffer.getvalue()
        ensure_public_text(public, destination.name)
        destination.write_text(public, newline="", encoding="utf-8")
    elif suffix in {".svg", ".npz"}:
        raw = public_npz(source) if suffix == ".npz" else public_svg(source, source.read_bytes())
        if suffix == ".svg":
            ensure_public_text(raw.decode("utf-8-sig"), destination.name)
        destination.write_bytes(raw)
    else:
        raise ValueError(f"Unrecognized public evidence extension: {suffix}")


def public_receipt(directory, destination):
    project_path = directory / "project.json"
    chain_path = directory / "chain.json"
    project = read_json(project_path)
    chain = read_json(chain_path)
    fields = {key: project[key] for key in RECEIPT_FIELDS if key in project}
    fields["gates"] = [{key: entry[key] for key in GATE_FIELDS if key in entry}
                      for entry in project.get("gates", ())]
    fields["receipt_chain"] = {key: chain[key] for key in CHAIN_FIELDS if key in chain}
    fields["local_evidence_sha256"] = {
        "project.json": digest(project_path),
        "chain.json": digest(chain_path),
    }
    fields["scope"] = "Filtered identifiers and statuses. Native store and traces remain local."
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, filtered_json(fields))
    ensure_public_text(destination.read_text(encoding="utf-8"), destination.name)


def adjacent_example(pilot_root, destination):
    raw = pilot_root / "study-lc/raw"
    with (raw / "reference_tight.in.csv").open(newline="", encoding="utf-8-sig") as stream:
        points = {row["id"]: row for row in csv.DictReader(stream)
                  if row["id"] in {"5754", "5755"}}
    rows = []
    for setting, file in (("standard", "uniform_3.out.csv"), ("tight", "reference_tight.out.csv")):
        with (raw / file).open(newline="", encoding="utf-8-sig") as stream:
            for record in csv.DictReader(stream):
                identity = record["id"]
                if identity in points:
                    point = points[identity]
                    rows.append(
                        (setting, identity, point["x"], point["y"], record["label"],
                         record["time"], record["min_r"], record["evals"], record["energy_error"])
                    )
    if len(rows) != 4 or {(row[1], row[4]) for row in rows} != {("5754", "4"), ("5755", "1")}:
        raise ValueError("The archived pilot data no longer supports the adjacent-cell example.")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("setting", "flat_grid_id", "x", "y", "label", "physical_event_time",
                         "minimum_separation", "force_evaluations", "relative_energy_error"))
        writer.writerows(rows)


def timing_example(path, destination):
    required = {
        "grid", "pair", "adaptive_seconds", "uniform_seconds", "adaptive_evals", "uniform_evals",
    }
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("The timing CSV is missing fields required for the documentation example.")
        row = next(
            (entry for entry in reader
             if entry.get("valid", "true").casefold() in {"1", "yes", "true"}
             and all(entry[key] for key in required)
             and float(entry["adaptive_seconds"]) > 0 and float(entry["uniform_seconds"]) > 0),
            None,
        )
    if row is None:
        message = (
            "The holdout recorded no complete valid timing pair. The timing log remains available "
            "in [timings.csv](timings.csv)."
        )
    else:
        a_work = int(row["adaptive_evals"])
        u_work = int(row["uniform_evals"])
        a_time = float(row["adaptive_seconds"])
        u_time = float(row["uniform_seconds"])
        gap = abs(a_work - u_work) / max(a_work, u_work) * 100
        message = (
            f"In grid {row['grid']}, pair {row['pair']}, adaptive used {a_work:,} force evaluations "
            f"and took {a_time:.3f} s. Uniform used {u_work:,} evaluations and took {u_time:.3f} s. "
            f"The work counts differed by {gap:.3f}%. This pair shows how similar force counts can "
            "have different wall time. Use the [full timing data](timings.csv) and the "
            "[held-out report](report.md) for the repeated comparison."
        )
    text = "# A measured timing pair\n\n" + message + "\n"
    ensure_public_text(text, destination.name)
    destination.write_text(text, encoding="utf-8")


def path_plan(workspace, pilot_root, holdout, staged):
    entries = [(workspace / name, staged / name, True) for name in SOURCE]
    for archive, public_name, filenames in PILOTS:
        entries.extend(
            (pilot_root / archive / name, staged / "evidence" / public_name / name, False)
            for name in filenames
        )
    entries.extend(
        (holdout / name, staged / "evidence/holdout" / name, False) for name in HOLDOUT
    )
    diagnostic = pilot_root / "extended-precision/diagnostic.json"
    if diagnostic.exists():
        entries.append((diagnostic, staged / "evidence/pilot-precision/binary128-diagnostic.json", False))
    return entries


def package(workspace, pilot_root, holdout, jovin, destination):
    if destination.exists():
        raise FileExistsError("The destination must be new.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=destination.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        entries = path_plan(workspace, pilot_root, holdout, staged)
        included = {target.resolve() for _, target, _ in entries}
        provenance = {}
        for source, target, original in entries:
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"Missing or unapproved release input: {source}")
            transfer(source, target, included, original=original)
            provenance[target.relative_to(staged).as_posix()] = {
                "original_sha256": digest(source),
                "public_sha256": digest(target),
            }
        evidence = staged / "evidence"
        if (pilot_root / "jovin-refinement-combined/project.json").is_file():
            public_receipt(pilot_root / "jovin-refinement-combined", evidence / "jovin-pilot.json")
        public_receipt(jovin, evidence / "jovin-holdout.json")
        adjacent_example(pilot_root, evidence / "pilot-lc/adjacent-example.csv")
        write_json(
            evidence / "pilot-lc/adjacent-example-provenance.json",
            {
                "note": "Example selected from original pilot rows 5754 and 5755.",
                "source_sha256": {
                    name: digest(pilot_root / "study-lc/raw" / name)
                    for name in ("reference_tight.in.csv", "reference_tight.out.csv", "uniform_3.out.csv")
                },
            },
        )
        timing_example(evidence / "holdout/timings.csv", evidence / "holdout/timing-example.md")
        (staged / "README.md").write_bytes((staged / "docs/public-readme.md").read_bytes())
        (staged / "NOTICE.md").write_bytes((staged / "docs/public-rights.md").read_bytes())
        (staged / "requirements.txt").write_text("numpy\nscipy\n", encoding="utf-8")
        (staged / ".gitattributes").write_text(
            "# Keep the original bytes of frozen sources and evidence.\n* -text\n", encoding="utf-8"
        )
        write_json(
            evidence / "published-provenance.json",
            {
                "note": "Original hashes identify source files or local evidence before filtering. "
                        "Public hashes verify the curated copies.",
                "files": provenance,
            },
        )
        files = sorted(
            (item for item in staged.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(staged).as_posix(),
        )
        checks = []
        for item in files:
            name = item.relative_to(staged).as_posix()
            ensure_public_text(name, name)
            if item.suffix != ".npz":
                ensure_public_text(item.read_text(encoding="utf-8-sig"), name)
            if item.stat().st_size > 50 * 1024 * 1024:
                raise ValueError(f"Public file exceeds the package size limit: {name}")
            checks.append(f"{digest(item)}  {name}")
        (staged / "SHA256SUMS").write_text("\n".join(checks) + "\n", encoding="utf-8")
        staged.rename(destination)
    return len(checks) + 1


def main():
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True, help="Finished holdout output.")
    parser.add_argument("--dest", type=Path, required=True, help="New staging directory for review.")
    parser.add_argument("--pilot-root", type=Path, default=workspace / "out",
                        help="Directory containing the archived study and study-lc runs.")
    parser.add_argument("--jovin", type=Path, default=workspace / "out/jovin-holdout",
                        help="Directory containing the holdout project and chain receipts.")
    args = parser.parse_args()
    destination = args.dest.resolve()
    try:
        number = package(workspace, args.pilot_root.resolve(), args.holdout.resolve(),
                         args.jovin.resolve(), destination)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Prepared {number} files for review in {destination}.")


if __name__ == "__main__":
    main()

