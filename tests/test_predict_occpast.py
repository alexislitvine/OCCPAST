import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from tqdm import tqdm

from predict_OCCPAST import (
    _discover_format_only_datasets,
    _explicit_format_only_dataset,
    _parse_csv_selection,
    _preprocess_dataset,
    _select_format_only_datasets,
    _validate_language_mode_args,
)


class TestPredictOCCPASTSelection(unittest.TestCase):
    def test_parse_csv_selection_accepts_comma_separated_indexes(self):
        self.assertEqual(_parse_csv_selection("1,3,5", 5), [1, 3, 5])

    def test_parse_csv_selection_accepts_comma_separated_indexes_with_spaces(self):
        self.assertEqual(_parse_csv_selection("1, 3, 5", 5), [1, 3, 5])

    def test_parse_csv_selection_accepts_ranges_and_deduplicates(self):
        self.assertEqual(_parse_csv_selection("1,3-5,4", 6), [1, 3, 4, 5])


class TestPredictOCCPASTFormatOnly(unittest.TestCase):
    def test_discovers_dataset_with_only_pst_prediction_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pst_csv = tmp_path / "jobs_predictions_pst_2026-06-08_120000_model.csv"
            pst_csv.write_text("id,occ1,pst_1\n1,Baker,\"1,2,3\"\n", encoding="utf-8")

            datasets = _discover_format_only_datasets(tmp_path, "both")

            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0].base, "jobs")
            self.assertIsNone(datasets[0].hisco_csv)
            self.assertEqual(datasets[0].pst_csv, pst_csv)

    def test_discovers_dataset_with_only_hisco_prediction_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hisco_csv = tmp_path / "jobs_predictions_hisco_2026-06-08_120000.csv"
            hisco_csv.write_text("id,occ1,hisco_1\n1,Baker,12345\n", encoding="utf-8")

            datasets = _discover_format_only_datasets(tmp_path, "both")

            self.assertEqual(len(datasets), 1)
            self.assertEqual(datasets[0].hisco_csv, hisco_csv)
            self.assertIsNone(datasets[0].pst_csv)

    def test_discovers_multiple_datasets_and_selects_requested_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "jobs_predictions_pst_2026-06-08_120000_model.csv").write_text("", encoding="utf-8")
            (tmp_path / "miners_predictions_hisco_2026-06-08_120000.csv").write_text("", encoding="utf-8")
            (tmp_path / "tailors_predictions_pst_2026-06-08_120000_model.csv").write_text("", encoding="utf-8")

            datasets = _discover_format_only_datasets(tmp_path, "both")

            with patch("builtins.input", return_value="1,3"):
                selected = _select_format_only_datasets(datasets)

            self.assertEqual([dataset.base for dataset in selected], ["jobs", "tailors"])

    def test_explicit_csv_paths_skip_dataset_discovery(self):
        args = SimpleNamespace(
            hisco_csv="/tmp/jobs_predictions_hisco_2026-06-08_120000.csv",
            pst_csv=None,
            predict_system="both",
        )

        dataset = _explicit_format_only_dataset(args)

        self.assertEqual(dataset.base, "jobs")
        self.assertEqual(dataset.hisco_csv, Path(args.hisco_csv))
        self.assertIsNone(dataset.pst_csv)


class TestPredictOCCPASTPreprocess(unittest.TestCase):
    def setUp(self):
        tqdm.pandas(desc="Cleaning strings")

    def test_prompts_for_prediction_column_when_occ1_original_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "jobs.csv"
            predicted_dir = tmp_path / "predicted"
            predicted_dir.mkdir()
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "job_title": ["Baker", "Smith, metal", ""],
                    "note": ["a", "b", "c"],
                }
            ).to_csv(csv_path, index=False)

            with patch("builtins.input", return_value="job_title"):
                df = _preprocess_dataset(
                    csv_path,
                    predicted_dir,
                    "jobs",
                    "2026-06-08_120000",
                    "utf-8",
                    chunksize=2,
                    language="en",
                )

            self.assertEqual(df["occ1_original"].tolist(), ["Baker", "Smith, metal"])
            self.assertEqual(df["occ1_clean"].tolist(), ["Baker", "Smith metal"])

    def test_uses_requested_prediction_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "jobs.csv"
            predicted_dir = tmp_path / "predicted"
            predicted_dir.mkdir()
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "job_title": ["Tailor", "Miner"],
                }
            ).to_csv(csv_path, index=False)

            df = _preprocess_dataset(
                csv_path,
                predicted_dir,
                "jobs",
                "2026-06-08_120000",
                "utf-8",
                chunksize=2,
                prediction_column="job_title",
                language="en",
            )

            self.assertEqual(df["occ1_original"].tolist(), ["Tailor", "Miner"])

    def test_multilingual_mode_uses_language_column_and_unk_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "jobs.csv"
            predicted_dir = tmp_path / "predicted"
            predicted_dir.mkdir()
            pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "occ1_original": ["Tailor", "Miner", "Baker", "Teacher"],
                    "lang_col": ["en", "", None, "fr"],
                }
            ).to_csv(csv_path, index=False)

            df = _preprocess_dataset(
                csv_path,
                predicted_dir,
                "jobs",
                "2026-06-08_120000",
                "utf-8",
                chunksize=2,
                language_column="lang_col",
                unk_lang="unk",
            )

            self.assertEqual(df["lang"].tolist(), ["en", "unk", "unk", "fr"])

    def test_deduplicates_by_occ_and_language_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "jobs.csv"
            predicted_dir = tmp_path / "predicted"
            predicted_dir.mkdir()
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "occ1_original": ["Baker", "Baker", "Baker"],
                    "lang_col": ["en", "fr", "en"],
                }
            ).to_csv(csv_path, index=False)

            df = _preprocess_dataset(
                csv_path,
                predicted_dir,
                "jobs",
                "2026-06-08_120000",
                "utf-8",
                chunksize=3,
                language_column="lang_col",
                unk_lang="unk",
            )

            self.assertEqual(len(df), 2)
            self.assertEqual(sorted(df["lang"].tolist()), ["en", "fr"])


class TestPredictOCCPASTLanguageArgs(unittest.TestCase):
    def test_requires_exactly_one_language_mode(self):
        args = SimpleNamespace(
            format_only=False,
            language=None,
            language_column=None,
            unk_lang="unk",
        )
        with self.assertRaises(ValueError):
            _validate_language_mode_args(args)

    def test_rejects_both_language_modes(self):
        args = SimpleNamespace(
            format_only=False,
            language="en",
            language_column="lang",
            unk_lang="unk",
        )
        with self.assertRaises(ValueError):
            _validate_language_mode_args(args)

    def test_accepts_single_language_mode(self):
        args = SimpleNamespace(
            format_only=False,
            language="EN",
            language_column=None,
            unk_lang="UNK",
        )
        _validate_language_mode_args(args)
        self.assertEqual(args.language, "en")
        self.assertEqual(args.unk_lang, "unk")

    def test_accepts_multilingual_mode(self):
        args = SimpleNamespace(
            format_only=False,
            language=None,
            language_column=" lang ",
            unk_lang="UNK",
        )
        _validate_language_mode_args(args)
        self.assertEqual(args.language_column, "lang")
        self.assertEqual(args.unk_lang, "unk")


if __name__ == "__main__":
    unittest.main()
