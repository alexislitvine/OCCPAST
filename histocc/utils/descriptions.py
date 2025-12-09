"""
Utility functions for loading and working with occupational code descriptions.

This module provides functions to load code descriptions from various sources,
including the built-in HISCO Key.csv file or custom user-provided files.

The descriptions can be used during training to provide additional context
about what each occupational code represents, which may help improve model
performance by contextualizing the predictions.

Example Usage
-------------
>>> from histocc.utils.descriptions import load_descriptions_from_csv
>>> descriptions = load_descriptions_from_csv('my_descriptions.csv', code_col='code', desc_col='description')
>>> print(descriptions.get('1,1,0,0,0,0,0,0'))  # Look up a PST2 code
"""

from __future__ import annotations

from typing import Dict, Union

import pandas as pd

from ..datasets import DATASETS


def load_hisco_descriptions(
    include_special_codes: bool = True,
) -> Dict[Union[str, int], str]:
    """
    Load HISCO code descriptions from the Key.csv file.

    Returns a dictionary mapping HISCO codes to their English text descriptions.
    Both string and integer keys are supported for convenience.

    Parameters
    ----------
    include_special_codes : bool, default=True
        Whether to include special codes (-3, -2, -1) in the returned mapping.
        These represent "Non work related title", "Source explicitly states
        that the person does not work", and "Missing, no title" respectively.

    Returns
    -------
    dict[Union[str, int], str]
        Dictionary mapping HISCO codes (both as strings and integers) to their
        English text descriptions.

    Examples
    --------
    >>> descriptions = load_hisco_descriptions()
    >>> descriptions['61110']
    'Farmer, General'
    >>> descriptions[61110]
    'Farmer, General'
    >>> descriptions[-1]
    'Missing, no title'
    """
    keys = DATASETS['keys']()

    descriptions = {}

    for _, row in keys.iterrows():
        hisco_code = row['hisco']
        description = row['en_hisco_text']

        # Skip special codes if not requested
        if not include_special_codes and hisco_code < 0:
            continue

        # Add both string and integer versions for convenience
        descriptions[str(hisco_code)] = description
        descriptions[int(hisco_code)] = description

    return descriptions


def load_descriptions_from_csv(
    filepath: str,
    code_col: str,
    desc_col: str,
    lang_col: str = None,
    encoding: str = 'utf-8',
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Load code descriptions from a CSV file.

    This function allows you to provide custom descriptions for any coding
    system (PST2, ISCO, custom codes, etc.) by specifying a CSV file with
    code-to-description mappings.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing codes and descriptions.
    code_col : str
        Name of the column containing the codes.
    desc_col : str
        Name of the column containing the descriptions.
    lang_col : str, optional
        Name of the column containing the language codes. If provided, returns
        a nested dictionary {code: {lang: description}} for multilingual support.
    encoding : str, default='utf-8'
        Encoding of the CSV file.

    Returns
    -------
    Dict[str, str] or Dict[str, Dict[str, str]]
        If lang_col is None: Dictionary mapping codes (as strings) to their descriptions.
        If lang_col is provided: Nested dictionary {code: {lang: description}}.

    Examples
    --------
    >>> # For single-language descriptions:
    >>> descriptions = load_descriptions_from_csv(
    ...     'pst2_descriptions.csv',
    ...     code_col='system_code',
    ...     desc_col='description'
    ... )
    >>> descriptions['1,1,0,0,0,0,0,0']
    'Agriculture and related occupations'

    >>> # For multilingual descriptions:
    >>> descriptions = load_descriptions_from_csv(
    ...     'pst2_descriptions.csv',
    ...     code_col='system_code',
    ...     desc_col='description',
    ...     lang_col='lang'
    ... )
    >>> descriptions['1,0,0,0,0,0,0,0']['en']
    'Primary Sector'
    >>> descriptions['1,0,0,0,0,0,0,0']['fr']
    'Secteur Primaire'
    """
    dtype_dict = {code_col: str, desc_col: str}
    if lang_col:
        dtype_dict[lang_col] = str

    df = pd.read_csv(filepath, encoding=encoding, dtype=dtype_dict)

    if lang_col is None:
        # Single-language mode
        descriptions = {}
        for _, row in df.iterrows():
            code = str(row[code_col])
            desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ''
            descriptions[code] = desc
        return descriptions
    else:
        # Multilingual mode - nested dict {code: {lang: description}}
        descriptions = {}
        for _, row in df.iterrows():
            code = str(row[code_col])
            lang = str(row[lang_col]) if pd.notna(row[lang_col]) else 'unk'
            desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ''

            if code not in descriptions:
                descriptions[code] = {}
            descriptions[code][lang] = desc

        return descriptions


def load_descriptions_from_dataframe(
    df: pd.DataFrame,
    code_col: str,
    desc_col: str,
    lang_col: str = None,
) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Load code descriptions from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing codes and descriptions.
    code_col : str
        Name of the column containing the codes.
    desc_col : str
        Name of the column containing the descriptions.
    lang_col : str, optional
        Name of the column containing the language codes. If provided, returns
        a nested dictionary {code: {lang: description}} for multilingual support.

    Returns
    -------
    Dict[str, str] or Dict[str, Dict[str, str]]
        If lang_col is None: Dictionary mapping codes (as strings) to their descriptions.
        If lang_col is provided: Nested dictionary {code: {lang: description}}.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'code': ['001', '002', '003'],
    ...     'description': ['Farmer', 'Teacher', 'Doctor']
    ... })
    >>> descriptions = load_descriptions_from_dataframe(df, 'code', 'description')

    >>> # For multilingual:
    >>> df = pd.DataFrame({
    ...     'code': ['001', '001'],
    ...     'lang': ['en', 'fr'],
    ...     'description': ['Farmer', 'Agriculteur']
    ... })
    >>> descriptions = load_descriptions_from_dataframe(df, 'code', 'description', 'lang')
    >>> descriptions['001']['en']
    'Farmer'
    """
    if lang_col is None:
        descriptions = {}
        for _, row in df.iterrows():
            code = str(row[code_col])
            desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ''
            descriptions[code] = desc
        return descriptions
    else:
        descriptions = {}
        for _, row in df.iterrows():
            code = str(row[code_col])
            lang = str(row[lang_col]) if pd.notna(row[lang_col]) else 'unk'
            desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ''

            if code not in descriptions:
                descriptions[code] = {}
            descriptions[code][lang] = desc

        return descriptions

    return descriptions


def load_descriptions_from_dict(
    mapping: dict,
) -> Dict[str, str]:
    """
    Normalize a dictionary of descriptions to use string keys.

    Parameters
    ----------
    mapping : dict
        Dictionary mapping codes to descriptions. Keys can be any type.

    Returns
    -------
    Dict[str, str]
        Dictionary with string keys.

    Examples
    --------
    >>> descriptions = load_descriptions_from_dict({
    ...     1: 'Farmer',
    ...     2: 'Teacher',
    ...     '3': 'Doctor'
    ... })
    """
    return {str(k): str(v) for k, v in mapping.items()}


def get_hisco_description(
    hisco_code: Union[str, int],
    descriptions: Dict[Union[str, int], str] = None,
    default: str = "Unknown occupation",
) -> str:
    """
    Get the English description for a specific HISCO code.

    Parameters
    ----------
    hisco_code : str or int
        The HISCO code to look up.
    descriptions : dict, optional
        Pre-loaded descriptions dictionary. If None, will load from Key.csv.
        For performance, consider pre-loading and reusing the dictionary.
    default : str, default="Unknown occupation"
        Value to return if the HISCO code is not found.

    Returns
    -------
    str
        The English description of the occupation.

    Examples
    --------
    >>> get_hisco_description('61110')
    'Farmer, General'
    >>> get_hisco_description(99999)
    'Unknown occupation'
    >>> get_hisco_description(99999, default='N/A')
    'N/A'
    """
    if descriptions is None:
        descriptions = load_hisco_descriptions()

    # Try both string and integer lookup
    if hisco_code in descriptions:
        return descriptions[hisco_code]

    # Try converting to the other type
    try:
        alt_key = str(hisco_code) if isinstance(hisco_code, int) else int(hisco_code)
        if alt_key in descriptions:
            return descriptions[alt_key]
    except (ValueError, TypeError):
        pass

    return default


def get_description(
    code: Union[str, int],
    descriptions: Dict[str, str],
    default: str = "",
) -> str:
    """
    Get the description for a specific code from a descriptions dictionary.

    This is a general-purpose lookup function that works with any coding system.

    Parameters
    ----------
    code : str or int
        The code to look up.
    descriptions : dict
        Dictionary mapping codes to descriptions.
    default : str, default=""
        Value to return if the code is not found.

    Returns
    -------
    str
        The description for the code.

    Examples
    --------
    >>> descriptions = {'001': 'Farmer', '002': 'Teacher'}
    >>> get_description('001', descriptions)
    'Farmer'
    >>> get_description('999', descriptions, default='Unknown')
    'Unknown'
    """
    code_str = str(code)

    if code_str in descriptions:
        return descriptions[code_str]

    return default


def create_code_to_description_mapping() -> dict[int, str]:
    """
    Create a mapping from the integer 'code' column to descriptions.

    This is useful when working with the model's internal code representation
    rather than HISCO codes directly.

    Returns
    -------
    dict[int, str]
        Dictionary mapping integer codes (as used by the model) to their
        English text descriptions.

    Examples
    --------
    >>> code_descriptions = create_code_to_description_mapping()
    >>> code_descriptions[0]  # Code 0 is HISCO -3
    'Non work related title'
    """
    keys = DATASETS['keys']()

    return dict(zip(keys['code'], keys['en_hisco_text']))


def add_descriptions_to_dataframe(
    df: pd.DataFrame,
    hisco_column: str = 'hisco_1',
    description_column: str = 'hisco_description',
    descriptions: dict = None,
) -> pd.DataFrame:
    """
    Add a description column to a DataFrame based on HISCO codes.

    This function adds a new column containing the English text description
    for each HISCO code in the specified column. This can be used to enrich
    training data with descriptive information about occupations.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to add descriptions to.
    hisco_column : str, default='hisco_1'
        The name of the column containing HISCO codes.
    description_column : str, default='hisco_description'
        The name for the new description column.
    descriptions : dict, optional
        Pre-loaded descriptions dictionary. If None, will load from Key.csv.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with the description column added.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'hisco_1': ['61110', '95120', '-1']})
    >>> df_with_desc = add_descriptions_to_dataframe(df)
    >>> df_with_desc['hisco_description'].tolist()
    ['Farmer, General', 'Bricklayer', 'Missing, no title']
    """
    if descriptions is None:
        descriptions = load_hisco_descriptions()

    df = df.copy()
    df[description_column] = df[hisco_column].apply(
        lambda x: get_hisco_description(x, descriptions)
    )

    return df


def format_input_with_description(
    occ_text: str,
    hisco_code: Union[str, int],
    descriptions: dict = None,
    format_template: str = "{occ} [DESC: {desc}]",
) -> str:
    """
    Format an occupational description with its HISCO code description.

    This can be used during training to augment the input with contextual
    information about the target classification, potentially helping the
    model learn better associations.

    Parameters
    ----------
    occ_text : str
        The original occupational description text.
    hisco_code : str or int
        The HISCO code associated with this occupation.
    descriptions : dict, optional
        Pre-loaded descriptions dictionary.
    format_template : str, default="{occ} [DESC: {desc}]"
        Template for formatting the combined output. Use {occ} for the
        original text and {desc} for the HISCO description.

    Returns
    -------
    str
        The formatted string combining original text and description.

    Examples
    --------
    >>> format_input_with_description("farmer", "61110")
    'farmer [DESC: Farmer, General]'
    >>> format_input_with_description(
    ...     "carpenter", "95410",
    ...     format_template="{occ} ({desc})"
    ... )
    'carpenter (Carpenter, General)'
    """
    if descriptions is None:
        descriptions = load_hisco_descriptions()

    desc = get_hisco_description(hisco_code, descriptions)

    return format_template.format(occ=occ_text, desc=desc)
