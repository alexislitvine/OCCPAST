from __future__ import annotations

import math

import pandas as pd


NULL_LIKE_STRINGS = {"", "nan", "none", "null"}


def clean_target_value(value: object, *, allow_question_mark: bool = True) -> str | None:
    """Canonicalize a target cell value used in train/eval/probe.

    Returns None for missing/null-like inputs (including string variants such as
    "nan", "NaN", "none", "null", and whitespace-only strings).
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
    text = str(value).strip()
    if text.lower() in NULL_LIKE_STRINGS:
        return None
    if not allow_question_mark and text == "?":
        return None
    return text


def prepare_target_columns(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for col in target_cols:
        cleaned[col] = cleaned[col].map(clean_target_value)
    return cleaned


def get_gold_num_codes_from_values(values: list[object]) -> int:
    num_codes = 0
    for value in values:
        normalized = clean_target_value(value)
        if normalized is None or normalized == "?":
            break
        num_codes += 1
    return num_codes

