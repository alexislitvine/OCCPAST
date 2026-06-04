import json
import tempfile
import unittest
from pathlib import Path

from format_preds import format_predictions, serialize_formatted_entries


class TestFormatPredictions(unittest.TestCase):
    def test_formats_pst_only_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pst_csv = tmp_path / "predictions_pst.csv"
            lookup_json = tmp_path / "PST2CodeDict.json"

            pst_csv.write_text(
                "occ1,conf,pst_1,pst_2\n"
                "unk[SEP]smith/baker,0.91,\"1,2,3,4,5\",0\n",
                encoding="utf-8",
            )
            lookup_json.write_text(
                json.dumps({"42": {"code": "1,2,3,4,5,0,0,0"}}),
                encoding="utf-8",
            )

            entries, stats = format_predictions(
                pst2_csv_path=pst_csv,
                pst2_lookup_json_path=lookup_json,
            )

            self.assertEqual(stats.total_predictions_processed, 1)
            self.assertEqual(stats.failures, 0)
            self.assertEqual(
                serialize_formatted_entries(entries),
                [
                    {
                        "string": "smith baker",
                        "predictions": [{"uris": ["pst2:42"], "conf": "0.91"}],
                    }
                ],
            )

    def test_requires_at_least_one_prediction_csv(self):
        with self.assertRaisesRegex(ValueError, "At least one prediction CSV"):
            format_predictions()


if __name__ == "__main__":
    unittest.main()
