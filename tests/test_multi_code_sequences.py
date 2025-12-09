"""
Tests for multi-code sequence handling in formatters.

This test verifies that the formatter correctly handles multi-code sequences 
(e.g., '12345&67890') and that the fix for _list_of_formatted_codes works correctly.

Note: Tests that require network access to initialize OccCANINE are skipped in 
offline environments.
"""

import unittest

import numpy as np

from histocc.formatter import hisco_blocky5


class TestMultiCodeSequencesFormatter(unittest.TestCase):
    """Tests for multi-code sequence handling in the formatter."""

    def setUp(self):
        """Initialize the HISCO formatter."""
        self.formatter = hisco_blocky5()
        self.code_len = 5  # HISCO codes are 5 digits

    def test_single_code_transform(self):
        """Test that single codes are transformed correctly."""
        single_code = '12345'
        result = self.formatter.transform_label(single_code)
        
        # Result should be: [BOS, tokens..., PAD..., EOS]
        # For single code: [BOS, 5 tokens, 20 PAD, EOS] = 27 total
        self.assertEqual(len(result), self.formatter.max_seq_len)
        
        # First element should be BOS (2)
        self.assertEqual(result[0], 2)
        
        # Last element should be EOS (3)
        self.assertEqual(result[-1], 3)
        
        # Tokens 1-5 should be the code, 6-25 should be PAD (1)
        code_tokens = result[1:6]
        self.assertEqual(len(code_tokens), 5)
        
        pad_tokens = result[6:-1]
        self.assertTrue(all(t == 1 for t in pad_tokens))

    def test_multi_code_transform(self):
        """Test that multi-code sequences are transformed correctly."""
        multi_code = '12345&67890'
        result = self.formatter.transform_label(multi_code)
        
        # Result should have both codes
        # First 5 tokens after BOS: first code
        # Next 5 tokens: second code
        # Rest: PAD
        
        self.assertEqual(len(result), self.formatter.max_seq_len)
        
        # Tokens 1-5 should be first code, 6-10 should be second code
        first_code_tokens = result[1:6]
        second_code_tokens = result[6:11]
        pad_tokens = result[11:-1]
        
        self.assertEqual(len(first_code_tokens), 5)
        self.assertEqual(len(second_code_tokens), 5)
        self.assertTrue(all(t == 1 for t in pad_tokens))

    def test_cycle_consistency_single_code(self):
        """Test that single codes can be round-tripped."""
        test_codes = ['12345', '67890', '00000', '99999']
        
        for code in test_codes:
            with self.subTest(code=code):
                transformed = self.formatter.transform_label(code)
                cleaned = self.formatter.clean_pred(transformed.astype(int))
                self.assertEqual(code, cleaned)

    def test_cycle_consistency_multi_code(self):
        """Test that multi-code sequences can be round-tripped."""
        test_codes = [
            '12345&67890',
            '67890&12345',
            '12345&67890&11111',
            '99999&00000&11111&22222',
        ]
        
        for code in test_codes:
            with self.subTest(code=code):
                transformed = self.formatter.transform_label(code)
                cleaned = self.formatter.clean_pred(transformed.astype(int))
                self.assertEqual(code, cleaned)

    def test_permutation_difference(self):
        """Test that permuted multi-code sequences produce different transforms."""
        code1 = '12345&67890'
        code2 = '67890&12345'
        
        result1 = self.formatter.transform_label(code1)
        result2 = self.formatter.transform_label(code2)
        
        # Both should have the same length
        self.assertEqual(len(result1), len(result2))
        
        # But they should NOT be identical
        self.assertFalse(
            np.array_equal(result1, result2),
            "Permuted multi-code sequences should produce different transforms"
        )

    def test_token_extraction_for_multi_code(self):
        """Test the fix logic: extracting correct number of tokens for multi-code sequences."""
        # This tests the logic that the _list_of_formatted_codes method should use
        sep_value = self.formatter.sep_value  # '&'
        
        test_cases = [
            ('12345', 1, 5),
            ('12345&67890', 2, 10),
            ('12345&67890&11111', 3, 15),
            ('12345&67890&11111&22222', 4, 20),
        ]
        
        for code_str, expected_num_codes, expected_tokens in test_cases:
            with self.subTest(code_str=code_str):
                # Count codes (this is what the fix does)
                num_codes = len(code_str.split(sep_value)) if sep_value else 1
                tokens_needed = num_codes * self.code_len
                
                self.assertEqual(num_codes, expected_num_codes)
                self.assertEqual(tokens_needed, expected_tokens)
                
                # Transform and extract
                transformed = self.formatter.transform_label(code_str)
                extracted = transformed[1:1+tokens_needed]
                
                self.assertEqual(len(extracted), expected_tokens)
                
                # Verify no PAD tokens in extracted portion (since we're extracting
                # exactly the right number of tokens)
                self.assertTrue(all(t != 1 for t in extracted))

    def test_multi_code_concatenation(self):
        """Verify that multi-code transforms are concatenations of individual code transforms."""
        multi_code = '12345&67890'
        single_1 = '12345'
        single_2 = '67890'
        
        multi_result = self.formatter.transform_label(multi_code)
        single_1_result = self.formatter.transform_label(single_1)
        single_2_result = self.formatter.transform_label(single_2)
        
        # Extract just the code tokens (excluding BOS, PAD, EOS)
        multi_code_tokens = multi_result[1:11]  # 2 codes * 5 = 10 tokens
        single_1_tokens = single_1_result[1:6]  # 5 tokens
        single_2_tokens = single_2_result[1:6]  # 5 tokens
        
        # Multi-code should be concatenation of singles
        expected = np.concatenate([single_1_tokens, single_2_tokens])
        np.testing.assert_array_equal(multi_code_tokens, expected)


if __name__ == '__main__':
    unittest.main()
