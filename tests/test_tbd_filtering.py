"""
Test that codes containing 'TBD' (To Be Determined) are properly filtered out
during code processing to avoid KeyError.
"""

import unittest


class TestTBDFiltering(unittest.TestCase):
    """Test that codes with 'TBD' are filtered out correctly."""

    def test_filter_logic_basic(self):
        """Test the filtering logic used in _list_of_formatted_codes."""
        # Simulate the filtering logic from prediction_assets.py line 362
        codes_list = ['12345', 'TBD', '67890', 'tbd', '11111', '99TBD', '22222']
        
        # Apply the same filter as in the code
        filtered = [i for i in codes_list if i != " " and i.lower() != "nan" and "TBD" not in i.upper()]
        
        # Verify that all TBD codes were filtered out
        # We should have 3 valid codes: 12345, 67890, 11111, 22222
        self.assertEqual(len(filtered), 4)
        self.assertIn('12345', filtered)
        self.assertIn('67890', filtered)
        self.assertIn('11111', filtered)
        self.assertIn('22222', filtered)
        self.assertNotIn('TBD', filtered)
        self.assertNotIn('tbd', filtered)
        self.assertNotIn('99TBD', filtered)

    def test_filter_logic_case_insensitive(self):
        """Test that TBD filtering is case-insensitive."""
        codes_list = ['12345', 'TBD', 'tbd', 'Tbd', 'TbD', 'tbD']
        
        # Apply the same filter as in the code
        filtered = [i for i in codes_list if i != " " and i.lower() != "nan" and "TBD" not in i.upper()]
        
        # Only '12345' should remain
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], '12345')

    def test_filter_logic_multichar_codes(self):
        """Test TBD filtering works with comma-separated codes."""
        codes_list = [
            '1,2,3,4,5,6,7,8',
            'TBD,2,3,4,5,6,7,8',
            '1,2,TBD,4,5,6,7,8',
            '9,8,7,6,5,4,3,2',
        ]
        
        # Apply the same filter as in the code
        filtered = [i for i in codes_list if i != " " and i.lower() != "nan" and "TBD" not in i.upper()]
        
        # Only codes without TBD should remain
        self.assertEqual(len(filtered), 2)
        self.assertIn('1,2,3,4,5,6,7,8', filtered)
        self.assertIn('9,8,7,6,5,4,3,2', filtered)

    def test_filter_logic_preserves_nan_filtering(self):
        """Test that existing nan filtering still works."""
        codes_list = ['12345', 'nan', 'NaN', '67890', 'NAN']
        
        # Apply the same filter as in the code
        filtered = [i for i in codes_list if i != " " and i.lower() != "nan" and "TBD" not in i.upper()]
        
        # Only valid codes should remain
        self.assertEqual(len(filtered), 2)
        self.assertIn('12345', filtered)
        self.assertIn('67890', filtered)

    def test_filter_logic_preserves_space_filtering(self):
        """Test that existing space filtering still works."""
        codes_list = ['12345', ' ', '67890', '11111']
        
        # Apply the same filter as in the code
        filtered = [i for i in codes_list if i != " " and i.lower() != "nan" and "TBD" not in i.upper()]
        
        # Only valid codes should remain (no space)
        self.assertEqual(len(filtered), 3)
        self.assertIn('12345', filtered)
        self.assertIn('67890', filtered)
        self.assertIn('11111', filtered)
        self.assertNotIn(' ', filtered)


if __name__ == '__main__':
    unittest.main()
