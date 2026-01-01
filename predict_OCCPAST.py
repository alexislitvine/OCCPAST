# run_predictions_and_format.py
from pathlib import Path
import argparse
import pandas as pd
import datetime
from tqdm import tqdm
from histocc import OccCANINE
import unicodedata
import re  # NEW
import os

def detect_encoding(p: Path) -> str:
    """
    Minimal, dependency-free probe. Tries common encodings in order.
    Returns the first that cleanly decodes.
    """
    candidates = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    data = p.read_bytes()
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

def select_csv_file(directory: Path) -> Path:
    directory = Path(directory)
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        raise SystemExit(1)
    print("Available CSV files:")
    for idx, file in enumerate(csv_files, start=1):
        print(f"{idx}. {file.name}")
    choice = int(input("Enter the number of the file to predict: "))
    return csv_files[choice - 1]


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
        help="Path to a CSV file to predict (non-interactive). If omitted, you'll be prompted to pick a file from --input-dir."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="predictions/to_predict",
        help="Directory containing CSV files to choose from when --input is not provided."
    )
    parser.add_argument(
        "--lookup",
        type=str,
        default="predictions/occpast/updatedPST2CodeDict.json",
        help="Path to updatedPST2CodeDict.json."
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
        default="/rds/user/adl38/hpc-work/OCCPAST/Data/models/",
        help="Root directory containing PST models in subfolders. Each model folder should contain a last.bin file."
    )
    args = parser.parse_args()

    tqdm.pandas(desc="Cleaning strings")

    if args.input:
        csv_file = Path(args.input)
        if not csv_file.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_file}")
        data_dir = csv_file.parent
    else:
        default_input_dir = Path(args.input_dir)
        user_dir = input(f"Enter the directory containing CSV files [{default_input_dir}]: ").strip()
        data_dir = Path(user_dir) if user_dir else default_input_dir
        csv_file = select_csv_file(data_dir)
    file_base = csv_file.stem

    predicted_dir = Path(args.output_dir) if args.output_dir else (data_dir.parent / "predicted")
    predicted_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    HOW_MANY_PREDS = 10
    chunksize = 10000
    EXCEL_FRIENDLY = True
    out_enc = "utf-8-sig" if EXCEL_FRIENDLY else "utf-8"

    print("Preprocessing data…")
    with open(csv_file, "r", encoding="latin1") as f:
        total_lines = sum(1 for _ in f) - 1

    source_enc = detect_encoding(csv_file)
    print(f"Detected input encoding: {source_enc}")

    reader = pd.read_csv(csv_file, chunksize=chunksize, encoding=source_enc)
    clean_chunks = []

    for chunk in tqdm(reader,
                      total=(total_lines // chunksize) + 1,
                      desc="Preprocessing chunks",
                      unit="chunk"):
        mask = (
            chunk["occ1_original"].notna()
            & chunk["occ1_original"].astype(str).str.strip().astype(bool)
        )
        chunk = chunk.loc[mask].copy()
        clean_chunks.append(chunk)

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

    # ✅ Save cleaned + deduplicated copy
    cleaned_csv_out = predicted_dir / f"{file_base}_cleaned_{ts}.csv"
    df.to_csv(cleaned_csv_out, index=False, encoding=out_enc)
    print(f"→ Saved cleaned & deduplicated CSV: {cleaned_csv_out.name}")

    if len(df["id"].unique()) != len(df):
        raise ValueError("Non unique ids after preprocessing!")

    # --- run predictions on df (same as before) ---
    mod_hisco = OccCANINE(verbose=True)

    # Discover PST models with last.bin under model_root and select
    model_root = Path(args.model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    candidates: list[tuple[str, Path, float]] = []  # (name, bin_path, mtime)
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
    # Sort candidates by mtime desc
    candidates.sort(key=lambda x: x[2], reverse=True)
    for i, (name, bin_path, mtime) in enumerate(candidates, start=1):
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i}. {name}  [last.bin saved: {ts}] -> {bin_path}")

    while True:
        try:
            sel = input("Select PST model number (default 1): ").strip()
            idx = 1 if sel == "" else int(sel)
            if 1 <= idx <= len(candidates):
                break
            else:
                print(f"Please enter a number between 1 and {len(candidates)}.")
        except ValueError:
            print("Please enter a valid number.")

    chosen_name, chosen_bin, chosen_mtime = candidates[idx - 1]
    print(f"Using PST model: {chosen_name} ({chosen_bin})")

    mod_pst = OccCANINE(
        str(chosen_bin),
        hf=False,
        system="pst",
        use_within_block_sep=True,
        verbose=True,
    )

    print("Running HISCO predictions…")
    pred_hisco = mod_hisco(
        df.occ1_clean.tolist(),
        k_pred=HOW_MANY_PREDS,
        debug=args.debug,
    )
    pred_hisco["id"] = df["id"].tolist()
    pred_hisco["occ1"] = df["occ1_original"].tolist()
    pred_hisco["occ1_clean"] = df["occ1_clean"].tolist()
    hisco_out = predicted_dir / f"{file_base}_predictions_hisco_{ts}.csv"
    pred_hisco.to_csv(hisco_out, index=False, encoding=out_enc)
    print(f"→ Saved HISCO to {hisco_out.name}")

    print("Running PST predictions…")
    pred_pst = mod_pst(
        df.occ1_clean.tolist(),
        k_pred=HOW_MANY_PREDS,
        debug=args.debug,
    )
    pred_pst["id"] = df["id"].tolist()
    pred_pst["occ1"] = df["occ1_original"].tolist()
    pred_pst["occ1_clean"] = df["occ1_clean"].tolist()
    pst_out = predicted_dir / f"{file_base}_predictions_pst_{ts}.csv"
    pred_pst.to_csv(pst_out, index=False, encoding=out_enc)
    print(f"→ Saved PST   to {pst_out.name}")

    # 4) path to PST2 lookup json
    #    prompt so you can point to the exact file you want
    default_lookup = Path("predictions/occpast/updatedPST2CodeDict.json")
    user_lookup = input(f"Path to updatedPST2CodeDict.json [{default_lookup}]: ").strip()
    lookup_path = Path(args.lookup) if args.lookup else default_lookup
    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup not found: {lookup_path}")

    # 5) Format & merge predictions -> combined JSON (with progress bars)
    print("Formatting/merging predictions…")
    entries, stats = format_predictions(
        hisco_csv_path=hisco_out,
        pst2_csv_path=pst_out,
        pst2_lookup_json_path=lookup_path,
        csv_encoding=out_enc,
    )

    # Log duplicates like the Node script
    if stats.duplicate_strings:
        print("Duplicate entries found for the following strings:")
        for s, c in stats.duplicate_strings:
            print(f'"{s}" occurs {c} times')
    else:
        print("No duplicate entries found.")

    combined_json = predicted_dir / f"{file_base}_processedPredictions_{ts}.json"
    write_json(combined_json, serialize_formatted_entries(entries))
    print(f"→ Wrote combined formatted JSON: {combined_json.name}")
    print(
        f"Total predictions processed: {stats.total_predictions_processed} | "
        f"Failures: {stats.failures}"
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
