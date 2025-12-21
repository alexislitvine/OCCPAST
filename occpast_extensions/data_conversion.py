"""
Utility functions for converting CSV data files to Parquet format for faster loading.

This module provides functions to convert training/validation CSV files to Parquet format,
which significantly improves data loading performance by:
- Reducing parsing overhead (columnar binary format vs text parsing)
- Better compression
- Faster I/O operations
- Type preservation
"""

from pathlib import Path
from typing import Union, List
import pandas as pd


def convert_csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Union[str, Path, None] = None,
    dtype_overrides: dict = None,
    converters: dict = None,
) -> Path:
    """
    Convert a single CSV file to Parquet format.
    
    Args:
        csv_path: Path to the CSV file to convert
        parquet_path: Output path for the Parquet file. If None, replaces .csv with .parquet
        dtype_overrides: Dictionary of column dtype overrides for reading CSV
        converters: Dictionary of converter functions for specific columns
        
    Returns:
        Path to the created Parquet file
    """
    csv_path = Path(csv_path)
    
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)
    
    # Create output directory if it doesn't exist
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {csv_path} to {parquet_path}...")
    
    # Read CSV with appropriate settings
    read_kwargs = {}
    if dtype_overrides:
        read_kwargs['dtype'] = dtype_overrides
    if converters:
        read_kwargs['converters'] = converters
    
    df = pd.read_csv(csv_path, **read_kwargs)
    
    # Write to Parquet with optimal settings
    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression='snappy',  # Good balance of speed and compression
        index=False,
    )
    
    # Report file sizes
    csv_size = csv_path.stat().st_size / (1024 * 1024)  # MB
    parquet_size = parquet_path.stat().st_size / (1024 * 1024)  # MB
    print(f"  CSV size: {csv_size:.2f} MB")
    print(f"  Parquet size: {parquet_size:.2f} MB")
    
    # Avoid division by zero - handle empty files gracefully
    if parquet_size > 0 and csv_size > 0:
        compression_ratio = csv_size / parquet_size
        print(f"  Compression ratio: {compression_ratio:.2f}x")
    else:
        print(f"  Compression ratio: N/A (empty file)")
    
    return parquet_path


def convert_directory_to_parquet(
    csv_dir: Union[str, Path],
    parquet_dir: Union[str, Path, None] = None,
    dtype_overrides: dict = None,
    converters: dict = None,
    pattern: str = "*.csv",
) -> List[Path]:
    """
    Convert all CSV files in a directory to Parquet format.
    
    Args:
        csv_dir: Directory containing CSV files
        parquet_dir: Output directory for Parquet files. If None, uses same directory
        dtype_overrides: Dictionary of column dtype overrides for reading CSV
        converters: Dictionary of converter functions for specific columns
        pattern: Glob pattern to match CSV files (default: "*.csv")
        
    Returns:
        List of paths to created Parquet files
    """
    csv_dir = Path(csv_dir)
    
    if parquet_dir is None:
        parquet_dir = csv_dir
    else:
        parquet_dir = Path(parquet_dir)
    
    csv_files = list(csv_dir.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir} matching pattern {pattern}")
        return []
    
    print(f"Found {len(csv_files)} CSV files to convert")
    
    parquet_files = []
    for csv_file in csv_files:
        parquet_file = parquet_dir / csv_file.with_suffix('.parquet').name
        converted_path = convert_csv_to_parquet(
            csv_file,
            parquet_file,
            dtype_overrides=dtype_overrides,
            converters=converters,
        )
        parquet_files.append(converted_path)
    
    print(f"\nSuccessfully converted {len(parquet_files)} files to Parquet format")
    return parquet_files


def get_hisco_dtype_overrides() -> dict:
    """
    Get dtype overrides for HISCO training data columns.
    
    Returns:
        Dictionary mapping column names to dtypes
    """
    return {
        'lang': str,
        'code1': str,
        'code2': str,
        'code3': str,
        'code4': str,
        'code5': str,
    }


def get_hisco_converters() -> dict:
    """
    Get converter functions for HISCO training data columns.
    
    Ensures 'occ1' column is always read as string, even if value looks like a number.
    
    Returns:
        Dictionary mapping column names to converter functions
    """
    return {
        'occ1': lambda x: x,  # Ensure occ1 is always treated as string
    }


def get_occ1950_dtype_overrides() -> dict:
    """
    Get dtype overrides for OCC1950 training data columns.
    
    Returns:
        Dictionary mapping column names to dtypes
    """
    return {
        'lang': str,
        'OCC1950_1': str,
        'OCC1950_2': str,
    }


def get_occ1950_converters() -> dict:
    """
    Get converter functions for OCC1950 training data columns.
    
    Returns:
        Dictionary mapping column names to converter functions
    """
    return {
        'occ1': lambda x: x,  # Ensure occ1 is always treated as string
    }
