"""Shared dataset naming helpers."""

from __future__ import annotations

import re
from typing import Iterable


LEGACY_DATABANK_DATASET_IDS = {
    "BF-C2DL-HSC": "ds1",
    "BF-C2DL-MuSC": "ds2",
    "DIC-C2DH-HeLa": "ds3",
}

CTC_DATASET_PATTERN = re.compile(r"(?:BF|DIC|Fluo|PhC)-[A-Z]\dD[HL]-[A-Za-z0-9+.-]+")


def dataset_id(dataset_name: str) -> str:
    """Return a stable databank identifier for a dataset name."""
    if dataset_name in LEGACY_DATABANK_DATASET_IDS:
        return LEGACY_DATABANK_DATASET_IDS[dataset_name]

    normalized = dataset_name.strip().replace("+", "plus")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", normalized).strip("-").lower()
    if not slug:
        raise ValueError("Dataset name must not be empty.")
    return slug


def infer_dataset_name_from_text(values: Iterable[object]) -> str:
    """Infer a CTC dataset name from paths, experiment names, or other text."""
    for value in values:
        if value is None:
            continue
        match = CTC_DATASET_PATTERN.search(str(value))
        if match:
            return match.group(0)
    return "unknown"
