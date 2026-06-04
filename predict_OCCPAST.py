# run_predictions_and_format.py
from pathlib import Path
import argparse
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import unicodedata
import re  # NEW
import os

MAX_PARALLEL_SYSTEMS = 2

def sanitize_filename_component(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return s or "unknown_model"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got: {value}")
    return parsed


def detect_encoding(p: Path, sample_size: int = 1024 * 1024) -> str:
    """
    Minimal, dependency-free probe. Tries common encodings in order.
    Returns the first that cleanly decodes.
    """
    candidates = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    with p.open("rb") as f:
        data = f.read(sample_size)
    for enc in candidates:
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"  # last-resort fallback

def normalize_series_nfc(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda x: unicodedata.normalize("NFC", x))

# cleaning function (remove commas, semicolons, colons, slashes, dots)
def clean_string(text: str) -> str:
    if text is None:
        return ""
    # Normalize to NFC first to avoid mixed accent forms
    s = unicodedata.normalize("NFC", str(text))
    # Remove , ; : . / and backslash \
    s = re.sub(r"[,\.;:/\\]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

# import the formatting module:
from format_preds import (
    format_predictions,
    serialize_formatted_entries,
    write_quarter_samples,
    write_json,
)

def _parse_csv_selection(selection: str, max_index: int) -> list[int]:
    selection = selection.strip().lower()
    if selection in {"all", "*"}:
        return list(range(1, max_index + 1))

    indexes: list[int] = []
    seen: set[int] = set()
    parts = re.split(r"[\s,]+", selection)
    for part in parts:
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            expanded = range(start, end + 1)
        else:
            expanded = [int(part)]

        for idx in expanded:
            if idx < 1 or idx > max_index:
                raise ValueError(f"Selection {idx} is outside the available range 1-{max_index}.")
            if idx not in seen:
                indexes.append(idx)
                seen.add(idx)

    if not indexes:
        raise ValueError("No CSV files selected.")
    return indexes


def select_csv_files(directory: Path) -> list[Path]:
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        raise SystemExit(1)
    print("Available CSV files:")
    for idx, file in enumerate(csv_files, start=1):
        print(f"{idx}. {file.name}")
    while True:
        choice = input("Enter file number(s) to predict, e.g. 1,3-5 or all: ").strip()
        try:
            selected = _parse_csv_selection(choice, len(csv_files))
            return [csv_files[idx - 1] for idx in selected]
        except ValueError as exc:
            print(exc)


def input_csv_files(input_arg: str) -> list[Path]:
    csv_files = [Path(part.strip()) for part in input_arg.split(",") if part.strip()]
    if not csv_files:
        raise ValueError("--input did not contain any CSV paths.")
    for csv_file in csv_files:
        if not csv_file.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_file}")
    return csv_files


def prediction_output_dir(csv_file: Path, output_dir: str | None) -> Path:
    return Path(output_dir) if output_dir else (csv_file.parent.parent / "predicted")


def _latest_prediction_csv(search_dir: Path, system: str) -> Path | None:
    pattern = f"*_predictions_{system}_*.csv"
    matches = list(search_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]

def _extract_prediction_metadata(path: Path, system: str) -> tuple[str | None, str | None]:
    m = re.match(
        rf".*_predictions_{re.escape(system)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{6}})(?:_(.+))?$",
        path.stem,
    )
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _select_pst_model(args) -> tuple[str | None, Path | None]:
    if args.predict_system not in {"both", "pst"}:
        return None, None

    if args.model_bin:
        chosen_bin = Path(args.model_bin)
        if not chosen_bin.exists():
            raise FileNotFoundError(f"PST model not found: {chosen_bin}")
        return chosen_bin.parent.name, chosen_bin

    model_root = Path(args.model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    candidates: list[tuple[str, Path, float]] = []
    for entry in model_root.iterdir():
        if entry.is_dir():
            bin_path = entry / "last.bin"
            if bin_path.exists():
                try:
                    mtime = bin_path.stat().st_mtime
                except Exception:
                    mtime = 0.0
                candidates.append((entry.name, bin_path, mtime))

    if not candidates:
        raise SystemExit(f"No models with last.bin found under {model_root}")

    print("Available PST models (with last.bin):")
    candidates.sort(key=lambda x: x[2], reverse=True)
    for i, (name, bin_path, mtime) in enumerate(candidates, start=1):
        saved_at = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i}. {name}  [last.bin saved: {saved_at}] -> {bin_path}")

    while True:
        try:
            sel = input("Select PST model number (default 1): ").strip()
            idx = 1 if sel == "" else int(sel)
            if 1 <= idx <= len(candidates):
                break
            print(f"Please enter a number between 1 and {len(candidates)}.")
        except ValueError:
            print("Please enter a valid number.")

    chosen_name, chosen_bin, _chosen_mtime = candidates[idx - 1]
    return chosen_name, chosen_bin


def _preprocess_dataset(
    csv_file: Path,
    predicted_dir: Path,
    file_base: str,
    ts: str,
    out_enc: str,
    chunksize: int,
) -> pd.DataFrame:
    print(f"Preprocessing data from {csv_file.name}…")

    source_enc = detect_encoding(csv_file)
    print(f"Detected input encoding: {source_enc}")

    reader = pd.read_csv(csv_file, chunksize=chunksize, encoding=source_enc)
    clean_chunks = []

    for chunk in tqdm(reader, desc="Preprocessing chunks", unit="chunk"):
        required_cols = {"id", "occ1_original"}
        missing_cols = required_cols - set(chunk.columns)
        if missing_cols:
            missing = ", ".join(sorted(missing_cols))
            raise ValueError(f"{csv_file} is missing required column(s): {missing}")

        mask = (
            chunk["occ1_original"].notna()
            & chunk["occ1_original"].astype(str).str.strip().astype(bool)
        )
        chunk = chunk.loc[mask].copy()
        if not chunk.empty:
            clean_chunks.append(chunk)

    if not clean_chunks:
        raise ValueError(f"No non-empty occ1_original values found in {csv_file}")

    df = pd.concat(clean_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["id"])

    # Normalize accents first
    df["occ1_original"] = normalize_series_nfc(df["occ1_original"])

    # Clean strings
    df["occ1_clean"] = df["occ1_original"].progress_map(clean_string)
    df = df[df["occ1_clean"].astype(str).str.strip().astype(bool)]

    # Deduplicate by cleaned string BEFORE saving and predicting
    before = len(df)
    df = df.drop_duplicates(subset=["occ1_clean"], keep="first")
    after = len(df)
    print(f"Removed {before - after} duplicate string(s) based on 'occ1_clean'.")

    cleaned_csv_out = predicted_dir / f"{file_base}_cleaned_{ts}.csv"
    df.to_csv(cleaned_csv_out, index=False, encoding=out_enc)
    print(f"→ Saved cleaned & deduplicated CSV: {cleaned_csv_out.name}")

    if len(df["id"].unique()) != len(df):
        raise ValueError("Non unique ids after preprocessing!")

    return df


def _run_system_predictions(
    system: str,
    model,
    occ1_clean: list[str],
    ids: list,
    occ1_original: list[str],
    out_dir: Path,
    file_base: str,
    ts: str,
    out_enc: str,
    k_pred: int,
    debug: bool,
    max_num_codes: int | None,
    pst_model_slug: str | None = None,
) -> tuple[str, Path]:
    print(f"Running {system.upper()} predictions…")
    preds = model(
        occ1_clean,
        k_pred=k_pred,
        debug=debug,
        max_num_codes=max_num_codes,
    )
    preds["id"] = ids
    preds["occ1"] = occ1_original
    preds["occ1_clean"] = occ1_clean

    if system == "hisco":
        out_path = out_dir / f"{file_base}_predictions_hisco_{ts}.csv"
    elif system == "pst":
        out_path = out_dir / f"{file_base}_predictions_pst_{ts}_{pst_model_slug}.csv"
    else:
        raise ValueError(f"Unknown prediction system: {system}")

    preds.to_csv(out_path, index=False, encoding=out_enc)
    print(f"→ Saved {system.upper()} to {out_path.name}")
    return system, out_path


def _predict_csv_file(
    csv_file: Path,
    args,
    mod_hisco,
    mod_pst,
    pst_model_slug: str | None,
    ts: str,
    out_enc: str,
    chunksize: int,
    k_pred: int,
) -> None:
    file_base = csv_file.stem
    predicted_dir = prediction_output_dir(csv_file, args.output_dir)
    predicted_dir.mkdir(parents=True, exist_ok=True)

    df = _preprocess_dataset(csv_file, predicted_dir, file_base, ts, out_enc, chunksize)

    hisco_out = None
    pst_out = None
    occ1_clean_values = df["occ1_clean"].tolist()
    id_values = df["id"].tolist()
    occ1_original_values = df["occ1_original"].tolist()

    if args.parallel_systems and mod_hisco is not None and mod_pst is not None:
        print("Running HISCO and PST predictions in parallel…")
        max_workers = min(args.parallel_workers, MAX_PARALLEL_SYSTEMS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_system = {
                executor.submit(
                    _run_system_predictions,
                    "hisco",
                    mod_hisco,
                    occ1_clean_values,
                    id_values,
                    occ1_original_values,
                    predicted_dir,
                    file_base,
                    ts,
                    out_enc,
                    k_pred,
                    args.debug,
                    args.max_num_codes,
                ): "hisco",
                executor.submit(
                    _run_system_predictions,
                    "pst",
                    mod_pst,
                    occ1_clean_values,
                    id_values,
                    occ1_original_values,
                    predicted_dir,
                    file_base,
                    ts,
                    out_enc,
                    k_pred,
                    args.debug,
                    args.max_num_codes,
                    pst_model_slug,
                ): "pst",
            }
            for future in as_completed(future_to_system):
                expected_system = future_to_system[future]
                try:
                    system_name, out_path = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Parallel inference failed for expected {expected_system.upper()} task "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
                if system_name == "hisco":
                    hisco_out = out_path
                else:
                    pst_out = out_path
    else:
        if mod_hisco is not None:
            _, hisco_out = _run_system_predictions(
                "hisco",
                mod_hisco,
                occ1_clean_values,
                id_values,
                occ1_original_values,
                predicted_dir,
                file_base,
                ts,
                out_enc,
                k_pred,
                args.debug,
                args.max_num_codes,
            )

        if mod_pst is not None:
            _, pst_out = _run_system_predictions(
                "pst",
                mod_pst,
                occ1_clean_values,
                id_values,
                occ1_original_values,
                predicted_dir,
                file_base,
                ts,
                out_enc,
                k_pred,
                args.debug,
                args.max_num_codes,
                pst_model_slug,
            )

    if hisco_out is not None or pst_out is not None:
        lookup_path = None
        if pst_out is not None:
            lookup_path = Path(args.lookup)
            if not lookup_path.exists():
                raise FileNotFoundError(f"Lookup not found: {lookup_path}")
        if args.predict_system == "both" and (hisco_out is None or pst_out is None):
            raise RuntimeError("Both prediction outputs are required to format combined results.")

        print("Formatting predictions…")
        entries, stats = format_predictions(
            hisco_csv_path=hisco_out,
            pst2_csv_path=pst_out,
            pst2_lookup_json_path=lookup_path,
            csv_encoding=out_enc,
        )

        if stats.duplicate_strings:
            print("Duplicate entries found for the following strings:")
            for s, c in stats.duplicate_strings:
                print(f'"{s}" occurs {c} times')
        else:
            print("No duplicate entries found.")

        model_suffix = pst_model_slug or args.predict_system or "unknown_model"
        combined_json = predicted_dir / f"{file_base}_processedPredictions_{ts}_{model_suffix}.json"
        write_json(combined_json, serialize_formatted_entries(entries))
        print(f"→ Wrote formatted JSON: {combined_json.name}")
        print(
            f"Total predictions processed: {stats.total_predictions_processed} | "
            f"Failures: {stats.failures}"
        )


def main():
    # CLI flags
    parser = argparse.ArgumentParser(
        description="Run OccCANINE predictions and format outputs"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw greedy outputs from the predictor",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to one CSV file, or comma-separated CSV paths, to predict. If omitted, you'll be prompted to pick file(s) from --input-dir."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/rds/user/adl38/hpc-work/OCCPAST2/Data/predictions/to_predict",
        help="Directory containing CSV files to choose from when --input is not provided."
    )
    parser.add_argument(
        "--lookup",
        type=str,
        default="/rds/user/adl38/hpc-work/OCCPAST2/Data/predictions/occpast/PST2CodeDict.json",
        help="Path to PST2CodeDict.json."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs. Defaults to sibling 'predicted' folder next to the input file."
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="/rds/user/adl38/hpc-work/OCCPAST2/Data/pst2",
        help="Root directory containing PST models in subfolders. Each model folder should contain a last.bin file."
    )
    parser.add_argument(
        "--disallow-pad-inside-block",
        action="store_true",
        default=False,
        help="Disallow PAD during greedy decoding inside code blocks (seq2seq inference)."
    )
    parser.add_argument(
        "--disallow-zero-at-block-start",
        action="store_true",
        default=False,
        help="Disallow predicting token '0' at the start of each block during greedy decoding."
    )
    parser.add_argument(
        "--max-num-codes",
        type=int,
        default=None,
        help="Override max_num_codes for greedy decoding during prediction."
    )
    parser.add_argument(
        "--predict-system",
        type=str,
        choices=["both", "pst", "hisco"],
        default="both",
        help="Which system(s) to predict: both (default), pst only, or hisco only.",
    )
    parser.add_argument(
        "--model-bin",
        type=str,
        default=None,
        help="Optional direct path to a PST model .bin file. If set, skips interactive model selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=256,
        help="Batch size used during model inference (must be > 0).",
    )
    parser.add_argument(
        "--parallel-systems",
        action="store_true",
        help="Run HISCO and PST inference concurrently when --predict-system both is selected (GPU-bound workloads recommended).",
    )
    parser.add_argument(
        "--parallel-workers",
        type=positive_int,
        default=2,
        help="Worker count for --parallel-systems (must be > 0, capped at 2 for HISCO+PST).",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Skip preprocessing/inference and only format existing HISCO/PST CSV predictions into final JSON.",
    )
    parser.add_argument(
        "--hisco-csv",
        type=str,
        default=None,
        help="Path to an existing HISCO predictions CSV (used with --format-only).",
    )
    parser.add_argument(
        "--pst-csv",
        type=str,
        default=None,
        help="Path to an existing PST predictions CSV (used with --format-only).",
    )
    args = parser.parse_args()

    tqdm.pandas(desc="Cleaning strings")

    # Format-only flow: merge existing prediction CSVs into final JSON without inference.
    if args.format_only:
        auto_search_dir = Path(args.output_dir) if args.output_dir else (Path(args.input_dir).parent / "predicted")

        hisco_out = Path(args.hisco_csv) if args.hisco_csv and args.predict_system in {"both", "hisco"} else None
        pst_out = Path(args.pst_csv) if args.pst_csv and args.predict_system in {"both", "pst"} else None

        if hisco_out is None and args.predict_system in {"both", "hisco"}:
            hisco_out = _latest_prediction_csv(auto_search_dir, "hisco")
            if hisco_out is not None:
                print(f"Auto-selected latest HISCO CSV: {hisco_out}")
        if pst_out is None and args.predict_system in {"both", "pst"}:
            pst_out = _latest_prediction_csv(auto_search_dir, "pst")
            if pst_out is not None:
                print(f"Auto-selected latest PST CSV: {pst_out}")

        if hisco_out is None and args.predict_system in {"both", "hisco"}:
            raise FileNotFoundError(
                f"No HISCO prediction CSV found in {auto_search_dir}. Provide --hisco-csv explicitly."
            )
        if pst_out is None and args.predict_system in {"both", "pst"}:
            raise FileNotFoundError(
                f"No PST prediction CSV found in {auto_search_dir}. Provide --pst-csv explicitly."
            )

        if hisco_out is not None and not hisco_out.exists():
            raise FileNotFoundError(f"HISCO CSV not found: {hisco_out}")
        if pst_out is not None and not pst_out.exists():
            raise FileNotFoundError(f"PST CSV not found: {pst_out}")

        lookup_path = None
        if pst_out is not None:
            lookup_path = Path(args.lookup)
            if not lookup_path.exists():
                raise FileNotFoundError(f"Lookup not found: {lookup_path}")

        source_out = pst_out or hisco_out
        if source_out is None:
            raise RuntimeError("No prediction CSV selected for formatting.")

        predicted_dir = Path(args.output_dir) if args.output_dir else source_out.parent
        predicted_dir.mkdir(parents=True, exist_ok=True)

        ts_from_csv, model_from_csv = (None, None)
        if pst_out is not None:
            ts_from_csv, model_from_csv = _extract_prediction_metadata(pst_out, "pst")
        if ts_from_csv is None and hisco_out is not None:
            ts_from_csv, hisco_model_from_csv = _extract_prediction_metadata(hisco_out, "hisco")
            model_from_csv = model_from_csv or hisco_model_from_csv
        ts = ts_from_csv or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        model_suffix = sanitize_filename_component(model_from_csv or args.predict_system or "unknown_model")
        file_base = re.sub(r"_predictions_(hisco|pst)_.*$", "", source_out.stem)

        print("Formatting predictions…")
        entries, stats = format_predictions(
            hisco_csv_path=hisco_out,
            pst2_csv_path=pst_out,
            pst2_lookup_json_path=lookup_path,
            csv_encoding="utf-8-sig",
        )

        if stats.duplicate_strings:
            print("Duplicate entries found for the following strings:")
            for s, c in stats.duplicate_strings:
                print(f'"{s}" occurs {c} times')
        else:
            print("No duplicate entries found.")

        combined_json = predicted_dir / f"{file_base}_processedPredictions_{ts}_{model_suffix}.json"
        write_json(combined_json, serialize_formatted_entries(entries))
        print(f"→ Wrote formatted JSON: {combined_json.name}")
        print(
            f"Total predictions processed: {stats.total_predictions_processed} | "
            f"Failures: {stats.failures}"
        )
        print("All done ✅")
        return

    if args.input:
        csv_files = input_csv_files(args.input)
    else:
        default_input_dir = Path(args.input_dir)
        user_dir = input(f"Enter the directory containing CSV files [{default_input_dir}]: ").strip()
        data_dir = Path(user_dir) if user_dir else default_input_dir
        csv_files = select_csv_files(data_dir)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    HOW_MANY_PREDS = 10
    chunksize = 10000
    EXCEL_FRIENDLY = True
    out_enc = "utf-8-sig" if EXCEL_FRIENDLY else "utf-8"

    chosen_name, chosen_bin = _select_pst_model(args)

    # Import here to avoid slow startup before prompting the user.
    from histocc.prediction_assets import OccCANINE

    mod_hisco = None
    if args.predict_system in {"both", "hisco"}:
        mod_hisco = OccCANINE(
            verbose=True,
            batch_size=args.batch_size,
            disallow_pad_inside_block=args.disallow_pad_inside_block,
            disallow_zero_at_block_start=args.disallow_zero_at_block_start,
        )

    mod_pst = None
    pst_model_slug = None
    if chosen_bin is not None:
        print(f"Using PST model: {chosen_name} ({chosen_bin})")
        pst_model_slug = sanitize_filename_component(chosen_name)
        mod_pst = OccCANINE(
            str(chosen_bin),
            hf=False,
            system="pst",
            batch_size=args.batch_size,
            use_within_block_sep=True,
            verbose=True,
            disallow_pad_inside_block=args.disallow_pad_inside_block,
            disallow_zero_at_block_start=args.disallow_zero_at_block_start,
        )

    for idx, csv_file in enumerate(csv_files, start=1):
        print(f"\n=== Dataset {idx}/{len(csv_files)}: {csv_file.name} ===")
        _predict_csv_file(
            csv_file=csv_file,
            args=args,
            mod_hisco=mod_hisco,
            mod_pst=mod_pst,
            pst_model_slug=pst_model_slug,
            ts=ts,
            out_enc=out_enc,
            chunksize=chunksize,
            k_pred=HOW_MANY_PREDS,
        )

    # # 6) Create 4 sampled chunks as JSON + CSV (titles) beside the combined JSON
    # print("Writing 4 sampled quarter-chunks (JSON + CSV)…")
    # write_quarter_samples(
    #     formatted_entries=entries,
    #     out_dir=predicted_dir,
    #     base_name=f"{file_base}_titles",
    #     sample_size=300,
    #     seed=42,
    #     csv_encoding=out_enc,
    # )
    print("All done ✅")

if __name__ == "__main__":
    main()
