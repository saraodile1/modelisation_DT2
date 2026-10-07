from __future__ import annotations

"""Fonctions d'entree/sortie communes.

Centraliser la lecture/ecriture evite que chaque module gere differemment les
formats Excel, CSV, JSON ou texte.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    # Le pipeline accepte les fichiers Excel et CSV pour faciliter l'adaptation
    # a d'autres datasets.
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}. Use .xlsx, .xls or .csv.")


def write_json(data: dict[str, Any], path: Path) -> None:
    # JSON lisible humainement, utile pour les rapports machine-verifiables.
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(text: str, path: Path) -> None:
    # Markdown ou texte simple pour les resumes d'execution.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
