from __future__ import annotations

"""Validation du schema du dataset structure.

Ce module empeche le pipeline de partir sur un fichier incompatible. Les colonnes
medicaments et biomarqueurs sont configurables pour que le code reste stable
quand un autre dataset ajoute des variables.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SchemaReport:
    # missing_required bloque l'execution; les listes de medicaments servent de
    # trace dans output_pipeline/reports/schema_report.json.
    missing_required: list[str]
    medication_columns: list[str]
    additional_medication_columns: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_required


def infer_medication_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    # Priorite a la liste explicite du YAML: c'est la solution la plus fiable pour
    # un autre dataset, surtout si de nouveaux biomarqueurs numeriques apparaissent.
    explicit = cfg["schema"].get("medication_columns") or []
    if explicit:
        return [c for c in explicit if c in df.columns]

    # Fallback: si aucune liste explicite n'est fournie, toute colonne non protegee
    # est consideree comme medicament. Pratique, mais moins robuste scientifiquement.
    clinical = set(cfg["schema"]["clinical_columns"])
    clinical.update(cfg["schema"].get("additional_biomarker_columns") or [])
    profile = cfg["schema"]["profile_column"]
    id_col = cfg["schema"].get("id_column")
    protected = clinical | {profile}
    if id_col:
        protected.add(id_col)
    return [c for c in df.columns if c not in protected and not c.endswith("__is_taken")]


def validate_structured_dataset(df: pd.DataFrame, cfg: dict) -> SchemaReport:
    # Colonnes minimales necessaires a la partie actuelle du memoire. On ne rend
    # pas toutes les colonnes obligatoires, car certains datasets peuvent ajouter
    # ou retirer des variables secondaires.
    required = [
        c for c in cfg["schema"]["clinical_columns"]
        if c in {
            "Age", "Sexe", "IMC", "TT", "PAS", "PAD", "HTA", "HTS/HTD", "Prot",
            "HDL", "Totc", "LDLc", "TG", "Non HDL", "PCSK9", "Glu",
            "Ins (µUI/mL)", "HOMAB%", "HOMAS%", "HOMA-IR", "Adipo HMW",
            "Ratio Tot-c/HDL", "Ratio LDL-c/HDL", "Ratio TG/HDL",
            "Ratio LogTG/HDL", "lu(mmol/l)", "Insu pmol/L",
        }
    ]
    required.append(cfg["schema"]["profile_column"])
    required.extend(cfg["schema"]["antidiabetic_columns"])

    missing = [c for c in required if c not in df.columns]
    medication_columns = infer_medication_columns(df, cfg)
    additional_medications = [
        c for c in medication_columns
        if c not in cfg["schema"]["antidiabetic_columns"] and "__" not in c
    ]
    return SchemaReport(
        missing_required=missing,
        medication_columns=medication_columns,
        additional_medication_columns=additional_medications,
    )
