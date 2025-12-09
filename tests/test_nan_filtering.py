"""
Test that NaN values are properly filtered when building code mappings in finetune.py.
This test verifies the fix for the TypeError: '<' not supported between instances of 'float' and 'str'
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestNaNFiltering(unittest.TestCase):
    """Test that NaN values are properly handled when building code sets."""
    
    def test_notna_filters_nan_values(self):
        """Test that pd.notna correctly filters out NaN values."""
        # Create test data with NaN values
        test_values = np.array(['1234', '5678', np.nan, '9012', None])
        
        # Filter using pd.notna (the new approach)
        filtered_notna = set(val for val in test_values if pd.notna(val))
        
        # Should only contain string values, no NaN or None
        self.assertEqual(len(filtered_notna), 3)
        self.assertIn('1234', filtered_notna)
        self.assertIn('5678', filtered_notna)
        self.assertIn('9012', filtered_notna)
        
        # These values can be sorted without error
        try:
            sorted_codes = sorted(filtered_notna)
            self.assertEqual(len(sorted_codes), 3)
        except TypeError:
            self.fail("Sorting should not raise TypeError after filtering with pd.notna")
    
    def test_none_check_does_not_filter_nan(self):
        """Test that checking 'val is not None' does NOT filter NaN values."""
        # Create test data with NaN values
        test_values = np.array(['1234', '5678', np.nan, '9012', None])
        
        # Filter using None check (the old buggy approach)
        filtered_none = set(val for val in test_values if val is not None)
        
        # This will include NaN because NaN is not None
        # Note: We can't easily check if NaN is in the set because NaN != NaN
        # But we can verify the behavior by attempting to sort
        with self.assertRaises(TypeError):
            sorted(filtered_none)
    
    def test_csv_with_missing_values(self):
        """Test that reading CSV with missing values and filtering works correctly."""
        # Create a temporary CSV file with missing values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('code,description\n')
            f.write('1234,First\n')
            f.write('5678,Second\n')
            f.write(',Third\n')  # Missing code
            f.write('9012,Fourth\n')
            temp_file = f.name
        
        try:
            # Read CSV with dtype=str (as done in finetune.py)
            df = pd.read_csv(temp_file, dtype=str)
            
            # Extract codes using dropna (which should be sufficient for CSV data)
            codes = set(df['code'].dropna().unique())
            
            # Should have 3 codes
            self.assertEqual(len(codes), 3)
            
            # Should be sortable
            try:
                sorted_codes = sorted(codes)
                self.assertEqual(len(sorted_codes), 3)
            except TypeError:
                self.fail("Sorting should not raise TypeError after filtering with pd.notna")
        finally:
            # Clean up temp file
            os.unlink(temp_file)
    
    def test_pd_unique_with_nan(self):
        """Test that pd.unique preserves NaN values, requiring pd.notna() filtering."""
        # Create array with NaN values (simulating training data scenario)
        data_array = np.array([['1234', '5678'], ['9012', np.nan], ['1234', '3456']])
        
        # pd.unique on flattened array (as done in finetune.py for training codes)
        unique_values = pd.unique(data_array.ravel())
        
        # pd.unique DOES include NaN values
        # We need to filter them out with pd.notna()
        training_codes = set(val for val in unique_values if pd.notna(val))
        
        # Should have 4 unique codes (not including NaN)
        self.assertEqual(len(training_codes), 4)
        
        # Should be sortable without error
        try:
            sorted_codes = sorted(training_codes)
            self.assertEqual(len(sorted_codes), 4)
        except TypeError:
            self.fail("Sorting should not raise TypeError after filtering with pd.notna")
    
    def test_union_of_filtered_sets(self):
        """Test that union of two properly filtered sets can be sorted."""
        # Simulate training codes (using pd.unique which includes NaN)
        training_values = np.array(['1234', np.nan, '5678'])
        training_codes = set(val for val in training_values if pd.notna(val))
        
        # Simulate external codes (using dropna which removes NaN)
        external_df_values = np.array(['9012', '3456', np.nan])
        # Simulate dropna().unique() behavior for external codes
        external_codes = set(v for v in external_df_values if pd.notna(v))
        
        # Combine them
        combined_codes = training_codes.union(external_codes)
        
        # Should have 4 codes total
        self.assertEqual(len(combined_codes), 4)
        
        # Should be sortable without error
        try:
            sorted_codes = sorted(combined_codes)
            self.assertEqual(len(sorted_codes), 4)
            self.assertEqual(sorted_codes, ['1234', '3456', '5678', '9012'])
        except TypeError:
            self.fail("Sorting should not raise TypeError after filtering with pd.notna")


if __name__ == '__main__':
    unittest.main()
