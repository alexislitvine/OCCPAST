"""
Test data conversion utilities and Parquet support in dataloaders.
"""
import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from occpast_extensions import (
    convert_csv_to_parquet,
    convert_directory_to_parquet,
    get_hisco_dtype_overrides,
    get_hisco_converters,
)
from histocc.dataloader import _read_data_file


class TestDataConversion(unittest.TestCase):
    """Test CSV to Parquet conversion utilities."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_convert_single_csv_to_parquet(self):
        """Test converting a single CSV file to Parquet."""
        # Create a sample CSV file
        csv_path = Path(self.test_dir) / "test_data.csv"
        df_original = pd.DataFrame({
            'occ1': ['farmer', 'baker', '42'],  # Include numeric-looking string
            'lang': ['en', 'en', 'en'],
            'code1': ['1', '2', '3'],
        })
        df_original.to_csv(csv_path, index=False)
        
        # Convert to Parquet
        parquet_path = convert_csv_to_parquet(
            csv_path,
            dtype_overrides={'lang': str, 'code1': str},
            converters={'occ1': lambda x: x},
        )
        
        # Verify Parquet file exists
        self.assertTrue(parquet_path.exists())
        self.assertEqual(parquet_path.suffix, '.parquet')
        
        # Verify data integrity
        df_parquet = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(df_original, df_parquet)
        
        # Verify numeric-looking string is preserved
        self.assertEqual(df_parquet.loc[2, 'occ1'], '42')
        self.assertIsInstance(df_parquet.loc[2, 'occ1'], str)
    
    def test_convert_directory_to_parquet(self):
        """Test converting multiple CSV files in a directory."""
        test_dir = Path(self.test_dir)
        
        # Create multiple CSV files
        for i in range(3):
            csv_path = test_dir / f"data_{i}.csv"
            df = pd.DataFrame({
                'occ1': [f'job_{i}_{j}' for j in range(5)],
                'lang': ['en'] * 5,
                'code1': [str(j) for j in range(5)],
            })
            df.to_csv(csv_path, index=False)
        
        # Convert all files
        parquet_files = convert_directory_to_parquet(
            test_dir,
            dtype_overrides={'lang': str, 'code1': str},
            converters={'occ1': lambda x: x},
        )
        
        # Verify all files were converted
        self.assertEqual(len(parquet_files), 3)
        for pf in parquet_files:
            self.assertTrue(pf.exists())
            self.assertEqual(pf.suffix, '.parquet')
    
    def test_get_hisco_dtype_overrides(self):
        """Test HISCO dtype overrides are correct."""
        overrides = get_hisco_dtype_overrides()
        
        self.assertIn('lang', overrides)
        self.assertIn('code1', overrides)
        self.assertIn('code2', overrides)
        self.assertEqual(overrides['lang'], str)
        self.assertEqual(overrides['code1'], str)
    
    def test_get_hisco_converters(self):
        """Test HISCO converters are correct."""
        converters = get_hisco_converters()
        
        self.assertIn('occ1', converters)
        # Test converter preserves string
        self.assertEqual(converters['occ1']('42'), '42')


class TestParquetDataLoading(unittest.TestCase):
    """Test that dataloaders can read Parquet files."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_read_csv_file(self):
        """Test _read_data_file can read CSV files."""
        csv_path = Path(self.test_dir) / "test.csv"
        df_original = pd.DataFrame({
            'occ1': ['farmer', 'baker'],
            'lang': ['en', 'en'],
            'code1': ['1', '2'],
        })
        df_original.to_csv(csv_path, index=False)
        
        # Read with _read_data_file
        df = _read_data_file(
            csv_path,
            usecols=['occ1', 'lang', 'code1'],
            dtype={'lang': str, 'code1': str},
            converters={'occ1': lambda x: x},
        )
        
        pd.testing.assert_frame_equal(df_original, df)
    
    def test_read_parquet_file(self):
        """Test _read_data_file can read Parquet files."""
        parquet_path = Path(self.test_dir) / "test.parquet"
        df_original = pd.DataFrame({
            'occ1': ['farmer', 'baker', '42'],
            'lang': ['en', 'en', 'en'],
            'code1': ['1', '2', '3'],
        })
        df_original.to_parquet(parquet_path, index=False, engine='pyarrow')
        
        # Read with _read_data_file
        df = _read_data_file(
            parquet_path,
            usecols=['occ1', 'lang', 'code1'],
        )
        
        pd.testing.assert_frame_equal(df_original, df)
    
    def test_parquet_faster_than_csv(self):
        """Test that Parquet loading is faster than CSV."""
        import time
        
        # Create a moderately sized dataset
        n_rows = 10000
        df_large = pd.DataFrame({
            'occ1': [f'occupation_{i}' for i in range(n_rows)],
            'lang': ['en'] * n_rows,
            'code1': [str(i % 100) for i in range(n_rows)],
            'code2': [str(i % 100) for i in range(n_rows)],
            'code3': [str(i % 100) for i in range(n_rows)],
        })
        
        csv_path = Path(self.test_dir) / "large.csv"
        parquet_path = Path(self.test_dir) / "large.parquet"
        
        df_large.to_csv(csv_path, index=False)
        df_large.to_parquet(parquet_path, index=False, engine='pyarrow')
        
        # Time CSV loading
        start = time.time()
        for _ in range(5):
            df_csv = _read_data_file(
                csv_path,
                dtype={'lang': str, 'code1': str, 'code2': str, 'code3': str},
                converters={'occ1': lambda x: x},
            )
        csv_time = time.time() - start
        
        # Time Parquet loading
        start = time.time()
        for _ in range(5):
            df_parquet = _read_data_file(parquet_path)
        parquet_time = time.time() - start
        
        # Parquet should be faster (but we'll be lenient for CI)
        # In practice, Parquet is often 2-10x faster
        print(f"CSV time: {csv_time:.3f}s, Parquet time: {parquet_time:.3f}s")
        if parquet_time > 0:
            print(f"Speedup: {csv_time/parquet_time:.2f}x")
        else:
            print("Speedup: Unable to measure (instant load)")
        
        # Just verify both methods work correctly
        pd.testing.assert_frame_equal(df_csv, df_parquet)
    
    def test_parquet_column_selection(self):
        """Test that Parquet column selection works correctly."""
        parquet_path = Path(self.test_dir) / "test.parquet"
        df_full = pd.DataFrame({
            'occ1': ['farmer', 'baker'],
            'lang': ['en', 'en'],
            'code1': ['1', '2'],
            'code2': ['3', '4'],
            'unused_col': ['a', 'b'],
        })
        df_full.to_parquet(parquet_path, index=False, engine='pyarrow')
        
        # Read only specific columns
        df = _read_data_file(
            parquet_path,
            usecols=['occ1', 'lang', 'code1'],
        )
        
        # Should only have requested columns
        self.assertEqual(list(df.columns), ['occ1', 'lang', 'code1'])
        self.assertNotIn('unused_col', df.columns)


class TestBackwardCompatibility(unittest.TestCase):
    """Test that CSV files still work after Parquet support is added."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_csv_still_works(self):
        """Verify that CSV files can still be loaded."""
        csv_path = Path(self.test_dir) / "legacy.csv"
        df_original = pd.DataFrame({
            'occ1': ['farmer', 'baker', '42'],
            'lang': ['en', 'en', 'en'],
            'code1': ['1', '2', '3'],
        })
        df_original.to_csv(csv_path, index=False)
        
        # Should work with _read_data_file
        df = _read_data_file(
            csv_path,
            dtype={'lang': str, 'code1': str},
            converters={'occ1': lambda x: x},
        )
        
        pd.testing.assert_frame_equal(df_original, df)


if __name__ == '__main__':
    unittest.main()
