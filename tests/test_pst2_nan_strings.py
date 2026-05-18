import unittest
import pandas as pd

from finetune import _is_present_pst2, prepare_target_cols
from histocc.formatter import construct_general_purpose_formatter


class TestPst2NanStrings(unittest.TestCase):
    def test_nan_strings_treated_as_missing(self):
        formatter = construct_general_purpose_formatter(
            block_size=8,
            target_cols=["pst2_1", "pst2_2"],
        )
        data = pd.DataFrame(
            {
                "pst2_1": ["1,2,3,4,5,6,7,8", "1,2,3,4,5,6,7,8"],
                "pst2_2": ["nan", "None"],
                "occ1": ["example", "example"],
                "lang": ["en", "en"],
            }
        )

        prepared = prepare_target_cols(
            data=data.copy(),
            formatter=formatter,
            drop_bad_rows=False,
        )

        self.assertTrue(pd.isna(prepared.loc[0, "pst2_2"]))
        self.assertTrue(pd.isna(prepared.loc[1, "pst2_2"]))
        self.assertFalse(_is_present_pst2(prepared.loc[0, "pst2_2"]))
        self.assertFalse(_is_present_pst2(prepared.loc[1, "pst2_2"]))


if __name__ == "__main__":
    unittest.main()
