# format preds for occ past
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import unicodedata


from tqdm import tqdm


@dataclass
class Prediction:
    uris: List[str]
    conf: str | float | int | None


@dataclass
class FormattedEntry:
    string: str
    predictions: List[Prediction]


@dataclass
class FormatStats:
    total_predictions_processed: int
    failures: int
    duplicate_strings: List[Tuple[str, int]]  # [(string, count), ...]


# -----------------------
# Utilities
# -----------------------
def _norm_key(k: str) -> str:
    # strip BOM on first column if present + trim spaces
    return k.replace("\ufeff", "").strip()

def read_csv_dicts(path: Path, encoding: str = "utf-8") -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        # normalize fieldnames in-place so DictReader uses clean keys
        if reader.fieldnames:
            reader.fieldnames = [_norm_key(fn) for fn in reader.fieldnames]
        for row in tqdm(reader, desc=f"Reading {path.name}", unit="row"):
            # ensure all keys are normalized (paranoia if csv module changes)
            row = {_norm_key(k): v for k, v in row.items()}
            rows.append(row)
    return rows

def _normalize_occ1(occ1: str) -> str:
    """
    Remove leading 'unk[SEP]', replace slashes with spaces, strip,
    and normalize to NFC so accents render consistently.
    """
    if occ1.startswith("unk[SEP]"):
        occ1 = occ1[len("unk[SEP]") :]
    occ1 = occ1.replace("/", " ").strip()
    return unicodedata.normalize("NFC", occ1)

def _normalize_pst_code(code: str) -> str:
    """
    Ensure the PST code has 8 comma-separated components by right-padding with ',0',
    exactly like the JS logic.
    """
    parts = code.split(",")
    while len(parts) < 8:
        parts.append("0")
    return ",".join(parts)


def _is_valid_val(val: str | None, scheme: str) -> bool:
    """
    Match the JS guards for whether a predicted code should be used.
    - PST: reject None, "nan", "0", "-1"
    - HISCO: reject None, "nan", "-1"
    """
    if val is None:
        return False
    s = str(val).strip()
    if s.lower() == "nan":
        return False
    if scheme == "pst":
        if s in {"0", "-1"}:
            return False
    else:  # hisco
        if s in {"-1"}:
            return False
    return True

def _escape_title_for_csv(title: str) -> str:
    """
    Only used if you want to build CSV rows manually.
    Prefer csv.writer in write_csv() which handles quoting.
    """
    t = title.replace('"', '""')
    if any(c in t for c in [",", '"', "\n", "\r"]):
        return f'"{t}"'
    return t


# -----------------------
# IO
# -----------------------

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(path: Path, titles: Iterable[str], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title"])
        for t in titles:
            writer.writerow([unicodedata.normalize("NFC", t)])

# -----------------------
# Core: PST2 lookup
# -----------------------

def load_pst2_code_lookup(path: Path) -> Dict[str, str]:
    """
    Load updatedPST2CodeDict.json, which is shaped like {id: {code: 'a,b,c,...'}} in your JS.
    Then invert to {code_string -> id} (exactly as in the JS reduce()).
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # raw looks like: { "<id>": {"code": "x,y,z,..."} }
    inverted: Dict[str, str] = {}
    for _id, entry in raw.items():
        code = entry.get("code")
        if isinstance(code, str) and code:
            inverted[code] = _id
    return inverted


# -----------------------
# Formatting (Node parity)
# -----------------------

def _process_entry(
    entry: dict,
    scheme: str,
    pst2_lookup: Dict[str, str] | None,
    formatted_entries: List[FormattedEntry],
    counters: Dict[str, int],
) -> None:
    """
    Mutates formatted_entries and updates counters.
    counters keys: "count", "failure_count"
    """
    occ1 = entry.get("occ1", "") or ""
    string_val = _normalize_occ1(str(occ1))
    conf_val = entry.get("conf")

    pred = Prediction(uris=[], conf=conf_val)

    if scheme == "pst":
        # need pst2_lookup
        if pst2_lookup is None:
            raise ValueError("pst2_lookup is required for PST processing")

        for i in range(5):
            code_prop = f"pst_{i+1}"
            if code_prop in entry and _is_valid_val(entry.get(code_prop), "pst"):
                counters["count"] += 1
                code = str(entry[code_prop])
                code = _normalize_pst_code(code)
                pst2_code = pst2_lookup.get(code)
                if pst2_code:
                    pred.uris.append("pst2:" + pst2_code)
                else:
                    counters["failure_count"] += 1

    else:  # hisco
        for i in range(5):
            code_prop = f"hisco_{i+1}"
            if code_prop in entry and _is_valid_val(entry.get(code_prop), "hisco"):
                pred.uris.append("hisco:" + str(entry[code_prop]))

    formatted = FormattedEntry(string=string_val, predictions=[pred])

    # Merge with existing by same string; avoid adding duplicate URIs list
    existing = next((e for e in formatted_entries if e.string == formatted.string), None)
    if existing:
        new_uris_join = "".join(pred.uris)
        has_same = any("".join(p.uris) == new_uris_join for p in existing.predictions)
        if not has_same:
            existing.predictions.extend(formatted.predictions)
    else:
        formatted_entries.append(formatted)


def _find_duplicates(entries: List[FormattedEntry]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for e in entries:
        counts[e.string] = counts.get(e.string, 0) + 1
    dupes = [(s, c) for s, c in counts.items() if c > 1]
    dupes.sort(key=lambda x: (-x[1], x[0]))
    return dupes


def format_predictions(
    hisco_csv_path: Path,
    pst2_csv_path: Path,
    pst2_lookup_json_path: Path,
    csv_encoding: str = "utf-8",
) -> Tuple[List[FormattedEntry], FormatStats]:
    raw_hisco = read_csv_dicts(hisco_csv_path, encoding=csv_encoding)
    raw_pst2 = read_csv_dicts(pst2_csv_path, encoding=csv_encoding)
    pst2_lookup = load_pst2_code_lookup(pst2_lookup_json_path)

    formatted: List[FormattedEntry] = []
    counters = {"count": 0, "failure_count": 0}

    for row in tqdm(raw_pst2, desc="Processing PST2 predictions", unit="row"):
        _process_entry(row, "pst", pst2_lookup, formatted, counters)

    for row in tqdm(raw_hisco, desc="Processing HISCO predictions", unit="row"):
        _process_entry(row, "hisco", pst2_lookup=None, formatted_entries=formatted, counters=counters)

    duplicates = _find_duplicates(formatted)

    stats = FormatStats(
        total_predictions_processed=counters["count"],
        failures=counters["failure_count"],
        duplicate_strings=duplicates,
    )
    return formatted, stats


# -----------------------
# Chunking & CSV export (Node parity)
# -----------------------

def split_into_quarters(seq: List[FormattedEntry]) -> List[List[FormattedEntry]]:
    """
    Split into 4 contiguous quarters (like the JS slicing).
    """
    total = len(seq)
    qsize = total // 4
    return [
        seq[:qsize],
        seq[qsize : 2 * qsize],
        seq[2 * qsize : 3 * qsize],
        seq[3 * qsize :],
    ]


def sample_from_each_quarter(
    quarters: List[List[FormattedEntry]],
    sample_size: int = 300,
    seed: int | None = 42,
) -> List[List[FormattedEntry]]:
    """
    Deterministic sampling (unless seed is None) of N entries from each quarter,
    matching your JS logic (shuffle then slice).
    """
    rng = random.Random(seed)
    samples: List[List[FormattedEntry]] = []
    for q in quarters:
        q_copy = q[:]  # non-destructive
        rng.shuffle(q_copy)
        samples.append(q_copy[:sample_size])
    return samples


def entries_to_titles(entries: Iterable[FormattedEntry]) -> List[str]:
    """
    Extract 'string' (title) from each entry.
    """
    return [e.string for e in entries]


def write_quarter_samples(
    formatted_entries: List[FormattedEntry],
    out_dir: Path,
    base_name: str,
    sample_size: int = 300,
    seed: int | None = 42,
    csv_encoding: str = "utf-8-sig",
) -> List[Tuple[Path, Path]]:
    """
    Create 4 sampled chunks as JSON + CSV, returning [(json_path, csv_path), ...].
    - JSON files contain the list of entries (dict-serialized).
    - CSV files contain a single 'title' column with the entry strings.

    This mirrors the second half of the Node script.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to plain dicts for JSON
    def to_plain(e: FormattedEntry) -> dict:
        return {
            "string": e.string,
            "predictions": [{"uris": p.uris, "conf": p.conf} for p in e.predictions],
        }

    quarters = split_into_quarters(formatted_entries)
    samples = sample_from_each_quarter(quarters, sample_size=sample_size, seed=seed)

    results: List[Tuple[Path, Path]] = []
    for i, sample in enumerate(samples, start=1):
        json_path = out_dir / f"{base_name}_chunk{i}.json"
        csv_path = out_dir / f"{base_name}_titles_chunk{i}.csv"

        # JSON
        json_data = [to_plain(e) for e in sample]
        write_json(json_path, json_data)

        # CSV
        titles = entries_to_titles(sample)
        write_csv(csv_path, titles)

        # progress logs
        unique_count = len(set(titles))
        tqdm.write(f"Quarter {i}: JSON and CSV written.")
        tqdm.write(f"Quarter {i}: Unique titles in CSV: {unique_count}")

        results.append((json_path, csv_path))

    return results


# -----------------------
# Optional: JSON round-trip helpers
# -----------------------

def serialize_formatted_entries(entries: List[FormattedEntry]) -> List[dict]:
    return [
        {
            "string": e.string,
            "predictions": [{"uris": p.uris, "conf": p.conf} for p in e.predictions],
        }
        for e in entries
    ]


def deserialize_formatted_entries(data: List[dict]) -> List[FormattedEntry]:
    out: List[FormattedEntry] = []
    for item in data:
        preds = [Prediction(uris=p.get("uris", []), conf=p.get("conf")) for p in item.get("predictions", [])]
        out.append(FormattedEntry(string=item.get("string", ""), predictions=preds))
    return out


# -----------------------
# Simple CLI (optional)
# -----------------------

def main_cli(
    hisco_csv: str,
    pst2_csv: str,
    pst2_lookup_json: str,
    output_formatted_json: str,
    chunk_out_dir: str | None = None,
    chunk_base_name: str = "census_below_preds",
    sample_size: int = 300,
    seed: int | None = 42,
) -> None:
    """
    Optional CLI entrypoint so you can also run this as a script if desired.

    Example:
      python -m format_latest_preds \
        --hisco ./dpd_clustered_4_7/predictions_hisco.csv \
        --pst2 ./half_finished_pst2_model_check/census_1851_1921_over1000_predictions_pst_2025-09-11_225114.csv \
        --lookup ./updatedPST2CodeDict.json \
        --out ./2025_12_9_census_over_processedPredictions.json \
        --chunks ./chunks --base census_below_preds --n 300
    """
    hisco_csv_path = Path(hisco_csv)
    pst2_csv_path = Path(pst2_csv)
    pst2_lookup_path = Path(pst2_lookup_json)
    out_json_path = Path(output_formatted_json)

    entries, stats = format_predictions(hisco_csv_path, pst2_csv_path, pst2_lookup_path)

    # Log duplicates (parity with Node's console prints)
    if stats.duplicate_strings:
        tqdm.write("Duplicate entries found for the following strings:")
        for s, c in stats.duplicate_strings:
            tqdm.write(f'"{s}" occurs {c} times')
    else:
        tqdm.write("No duplicate entries found.")

    write_json(out_json_path, serialize_formatted_entries(entries))

    tqdm.write("Data processing and export completed successfully.")
    tqdm.write(
        f"Total predictions processed: {stats.total_predictions_processed} "
        f"Failures: {stats.failures}"
    )

    if chunk_out_dir:
        write_quarter_samples(
            entries,
            out_dir=Path(chunk_out_dir),
            base_name=chunk_base_name,
            sample_size=sample_size,
            seed=seed,
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Format predictions and create quarter samples + CSVs")
    p.add_argument("--hisco", required=True, help="Path to predictions_hisco.csv")
    p.add_argument("--pst2", required=True, help="Path to predictions_pst2.csv")
    p.add_argument("--lookup", required=True, help="Path to updatedPST2CodeDict.json")
    p.add_argument("--out", required=True, help="Output JSON for formatted predictions")
    p.add_argument("--chunks", default=None, help="Directory to write quarter chunk files (optional)")
    p.add_argument("--base", default="census_below_preds", help="Base name for chunk files")
    p.add_argument("--n", type=int, default=300, help="Sample size per quarter")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling (or omit to be non-deterministic)")
    args = p.parse_args()

    main_cli(
        hisco_csv=args.hisco,
        pst2_csv=args.pst2,
        pst2_lookup_json=args.lookup,
        output_formatted_json=args.out,
        chunk_out_dir=args.chunks,
        chunk_base_name=args.base,
        sample_size=args.n,
        seed=args.seed,
    )
