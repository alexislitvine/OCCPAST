"""
OCCPAST-specific extensions to the histocc package.

This module contains functionality specific to the OCCPAST project that extends
the upstream histocc package from OccCANINE.
"""

from .data_conversion import (
    convert_csv_to_parquet,
    convert_directory_to_parquet,
    get_hisco_dtype_overrides,
    get_hisco_converters,
    get_occ1950_dtype_overrides,
    get_occ1950_converters,
)

from .descriptions import (
    load_hisco_descriptions,
    load_descriptions_from_csv,
    load_descriptions_from_dataframe,
    load_descriptions_from_dict,
    get_hisco_description,
    get_description,
    create_code_to_description_mapping,
    add_descriptions_to_dataframe,
    format_input_with_description,
)

__all__ = [
    'convert_csv_to_parquet',
    'convert_directory_to_parquet',
    'get_hisco_dtype_overrides',
    'get_hisco_converters',
    'get_occ1950_dtype_overrides',
    'get_occ1950_converters',
    'load_hisco_descriptions',
    'load_descriptions_from_csv',
    'load_descriptions_from_dataframe',
    'load_descriptions_from_dict',
    'get_hisco_description',
    'get_description',
    'create_code_to_description_mapping',
    'add_descriptions_to_dataframe',
    'format_input_with_description',
]
