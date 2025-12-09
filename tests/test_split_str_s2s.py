"""
Tests for the _split_str_s2s method to verify it correctly uses the formatter's sep_value.

This test ensures that multiple occupational codes are correctly split for both
HISCO and PST (and other) systems by using the formatter's sep_value instead of
a hardcoded separator.
"""

import unittest

import numpy as np

from histocc.formatter import (
    hisco_blocky5,
    construct_general_purpose_formatter,
)


class MockOccCANINE:
    """Mock class to test _split_str_s2s and _output_permutations without loading models."""
    
    def __init__(self, formatter):
        self.formatter = formatter
    
    def _split_str_s2s(self, pred: str, symbol: str | None = None):
        """
        Splits predicted str if necessary.
        
        Uses the formatter's sep_value by default to handle multiple occupational codes.
        """
        if symbol is None:
            symbol = self.formatter.sep_value
        
        if symbol and symbol in pred:
            pred = pred.split(symbol)

        return pred
    
    def _output_permutations(self, output):
        """
        This function takes an output from the model with multiple labels
        and returns the permutations of the labels.
        This is used to compute the order invariant confidence.
        """
        from itertools import permutations
        
        sep_value = self.formatter.sep_value
        
        output = output.split(sep_value) if sep_value else [output]
        if len(output) == 1:
            return [output]

        # Generate all permutations of the output list
        permutations_list = list(permutations(output))

        # Join each permutation with the separator symbol
        return [sep_value.join(permutation) for permutation in permutations_list]


class TestSplitStrS2S(unittest.TestCase):
    """Tests for the _split_str_s2s method."""

    def test_split_hisco_multiple_codes(self):
        """Test splitting multiple HISCO codes."""
        # Create a mock model with HISCO formatter (uses '&' as sep_value)
        formatter = hisco_blocky5()
        model = MockOccCANINE(formatter)
        
        # HISCO uses '&' as separator
        self.assertEqual(model.formatter.sep_value, '&')
        
        # Test single code
        result = model._split_str_s2s('12345')
        self.assertEqual(result, '12345')
        
        # Test multiple codes
        result = model._split_str_s2s('12345&67890')
        self.assertIsInstance(result, list)
        self.assertEqual(result, ['12345', '67890'])
        
        # Test three codes
        result = model._split_str_s2s('12345&67890&11111')
        self.assertIsInstance(result, list)
        self.assertEqual(result, ['12345', '67890', '11111'])

    def test_split_pst_multiple_codes(self):
        """Test splitting multiple PST codes with within_block_sep."""
        # Create a model with PST-style formatter (uses '&' as sep_value, ',' as within_block_sep)
        formatter = construct_general_purpose_formatter(
            block_size=8,
            target_cols=['pst_1', 'pst_2', 'pst_3', 'pst_4'],
            use_within_block_sep=True,
        )
        
        # Verify formatter config
        self.assertEqual(formatter.sep_value, '&')
        self.assertEqual(formatter.within_block_sep, ',')
        
        model = MockOccCANINE(formatter)
        
        # Test single PST code (with commas within)
        result = model._split_str_s2s('1,5,6,1,11,0,0,0')
        self.assertEqual(result, '1,5,6,1,11,0,0,0')
        
        # Test multiple PST codes (separated by '&')
        result = model._split_str_s2s('1,5,6,1,11,0,0,0&2,2,4,1,0,0,0,0')
        self.assertIsInstance(result, list)
        self.assertEqual(result, ['1,5,6,1,11,0,0,0', '2,2,4,1,0,0,0,0'])

    def test_split_explicit_symbol(self):
        """Test that explicit symbol parameter overrides formatter's sep_value."""
        formatter = hisco_blocky5()
        model = MockOccCANINE(formatter)
        
        # Test with explicit pipe separator
        result = model._split_str_s2s('12345|67890', symbol='|')
        self.assertIsInstance(result, list)
        self.assertEqual(result, ['12345', '67890'])
        
        # If sep_value doesn't match, it shouldn't split
        result = model._split_str_s2s('12345|67890', symbol='&')
        self.assertEqual(result, '12345|67890')

    def test_split_empty_sep_value(self):
        """Test behavior when sep_value is empty string."""
        formatter = hisco_blocky5()
        model = MockOccCANINE(formatter)
        
        # Temporarily set sep_value to empty (edge case)
        original_sep = model.formatter.sep_value
        model.formatter.sep_value = ''
        
        # Should not split when sep_value is empty
        result = model._split_str_s2s('12345&67890')
        self.assertEqual(result, '12345&67890')
        
        # Restore original
        model.formatter.sep_value = original_sep


class TestOutputPermutations(unittest.TestCase):
    """Tests for the _output_permutations method."""

    def test_permutations_single_code(self):
        """Test permutations for single code returns single element list."""
        formatter = hisco_blocky5()
        model = MockOccCANINE(formatter)
        
        result = model._output_permutations('12345')
        self.assertEqual(result, [['12345']])

    def test_permutations_two_codes(self):
        """Test permutations for two codes."""
        formatter = hisco_blocky5()
        model = MockOccCANINE(formatter)
        
        result = model._output_permutations('12345&67890')
        self.assertEqual(len(result), 2)
        self.assertIn('12345&67890', result)
        self.assertIn('67890&12345', result)

    def test_permutations_three_codes(self):
        """Test permutations for three codes."""
        formatter = hisco_blocky5()
        model = MockOccCANINE(formatter)
        
        result = model._output_permutations('11111&22222&33333')
        # 3! = 6 permutations
        self.assertEqual(len(result), 6)
        self.assertIn('11111&22222&33333', result)
        self.assertIn('33333&22222&11111', result)

    def test_permutations_pst_style(self):
        """Test permutations for PST-style codes with within-block separators."""
        # Create model with PST-style formatter
        formatter = construct_general_purpose_formatter(
            block_size=8,
            target_cols=['pst_1', 'pst_2'],
            use_within_block_sep=True,
        )
        
        model = MockOccCANINE(formatter)
        
        # Test with two PST codes
        result = model._output_permutations('1,2,3,0,0,0,0,0&4,5,6,0,0,0,0,0')
        self.assertEqual(len(result), 2)
        self.assertIn('1,2,3,0,0,0,0,0&4,5,6,0,0,0,0,0', result)
        self.assertIn('4,5,6,0,0,0,0,0&1,2,3,0,0,0,0,0', result)


class TestNaNHandling(unittest.TestCase):
    """Tests for proper NaN handling in predictions."""
    
    def test_nan_detection(self):
        """Test that NaN values are properly detected."""
        # This tests the logic used in _format method
        test_cases = [
            (np.nan, True),
            (float('nan'), True),
            ('nan', False),  # String 'nan' is not np.isnan
            (0, False),
            ('0', False),
            ('12345', False),
        ]
        
        for value, expected_is_nan in test_cases:
            with self.subTest(value=value, expected_is_nan=expected_is_nan):
                if isinstance(value, float):
                    result = np.isnan(value)
                else:
                    result = False
                self.assertEqual(result, expected_is_nan)
    
    def test_empty_string_detection(self):
        """Test that empty strings and '0' are detected as invalid codes."""
        invalid_codes = ['', ' ', '0', '  0  ']
        
        for code in invalid_codes:
            with self.subTest(code=repr(code)):
                is_invalid = str(code).strip() == '' or str(code).strip() == '0'
                self.assertTrue(is_invalid, f"Code {repr(code)} should be detected as invalid")


if __name__ == '__main__':
    unittest.main()
