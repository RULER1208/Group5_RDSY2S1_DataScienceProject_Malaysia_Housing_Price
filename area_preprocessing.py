from __future__ import annotations
import re
import unicodedata
import pandas as pd

PLACEHOLDERS = {"", "N/A", "NA", "NULL", "NONE", "NAN", "-"}

def _base_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def clean_state_name(value) -> str:
    text = _base_text(value)
    if text.upper() in PLACEHOLDERS:
        return "UNKNOWN STATE"
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    return text.upper()

def clean_area_name(value) -> str:
    text = _base_text(value)
    if text.upper() in PLACEHOLDERS:
        return "UNKNOWN AREA"
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*&\s*", " & ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()

def create_area_key(state, area) -> str:
    return f"{clean_state_name(state)} | {clean_area_name(area)}"

def display_name(value: str) -> str:
    if not value:
        return value
    protected = {"KL", "MRT", "LRT", "USJ", "TTDI", "PJ", "JB"}
    parts = []
    for token in str(value).split(" "):
        parts.append(token if token in protected else token.title())
    return " ".join(parts)
