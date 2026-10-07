from __future__ import annotations

"""Nettoyage analytique pre-imputation.

Ce module correspond au deuxieme niveau de nettoyage: il ne restructure pas les
traitements bruts, mais corrige les problemes decouverts pendant l'EDA et prepare
une table compatible avec l'imputation MissForest.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import infer_medication_columns


@dataclass
class CleaningResult:
    # Les trois tables sont conservees pour tracer le pipeline:
    # raw = entree structuree, cleaned_for_eda = apres correction des sentinelles,
    # pre_imputation = table finale envoyee a MissForest.
    raw: pd.DataFrame
    cleaned_for_eda: pd.DataFrame
    pre_imputation: pd.DataFrame
    medication_columns: list[str]
    excluded_columns: list[str]
    derived_columns: list[str]
    missingness_table: pd.DataFrame


def _replace_numeric_sentinels(df: pd.DataFrame, cfg: dict, medication_columns: list[str]) -> pd.DataFrame:
    # Remplacer les valeurs sentinelles par de vrais NaN est crucial: pandas,
    # MissForest et les statistiques de missingness ne voient pas "-1" ou "0"
    # comme du manquant si on ne le declare pas explicitement.
    out = df.copy()
    missing_cfg = cfg["cleaning"]["sentinel_missing"]
    med_sentinel = missing_cfg["medication_taken_unknown_dose"]

    if med_sentinel is not None:
        # Convention propre au dataset actuel: -1 = medicament pris, dose inconnue.
        # Pour un autre dataset, cette valeur est changee dans config/pipeline.yaml.
        for col in medication_columns:
            if col in out.columns:
                out[col] = out[col].replace(med_sentinel, np.nan)

    # Certains zeros sont physiologiquement impossibles dans ce contexte et
    # representent du manquant deguise, pas de vraies mesures.
    for col in missing_cfg["numeric_zero_as_missing"]:
        if col in out.columns:
            out[col] = out[col].replace(0, np.nan)

    # Les colonnes HOMA contenaient parfois du texte ("Unable to calculate");
    # errors="coerce" transforme ces textes en NaN pour l'imputation.
    for col in ["HOMAB%", "HOMAS%", "HOMA-IR"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    # Harmonisation minimale des categories observees dans le dataset.
    hta_map = cfg["cleaning"]["harmonization"].get("HTA", {})
    if "HTA" in out.columns:
        out["HTA"] = out["HTA"].replace(hta_map)

    hts_map = cfg["cleaning"]["harmonization"].get("HTS/HTD", {}).copy()
    bool_false_value = hts_map.pop("false_boolean", None)
    if "HTS/HTD" in out.columns:
        out["HTS/HTD"] = out["HTS/HTD"].replace(hts_map)
        if bool_false_value is not None:
            out["HTS/HTD"] = out["HTS/HTD"].replace({False: bool_false_value})

    return out


def prepare_cleaning(df: pd.DataFrame, cfg: dict) -> CleaningResult:
    # Les medicaments sont definis dans la config pour permettre a un autre
    # dataset d'avoir plus ou moins de colonnes sans modifier le code.
    medication_columns = infer_medication_columns(df, cfg)
    cleaned_for_eda = _replace_numeric_sentinels(df, cfg, medication_columns)

    # Colonnes techniques issues du nettoyage structurel: utiles pour audit, mais
    # non destinees a l'imputation ni au ML.
    helper_suffixes = cfg["cleaning"].get("helper_column_suffixes_to_drop", [])
    helper_columns = [
        c for c in cleaned_for_eda.columns
        if any(c.endswith(suffix) for suffix in helper_suffixes)
    ]
    helper_columns.extend([
        c for c in cfg["cleaning"].get("helper_columns_to_drop", [])
        if c in cleaned_for_eda.columns
    ])
    if helper_columns:
        cleaned_for_eda = cleaned_for_eda.drop(columns=sorted(set(helper_columns)))

    # Table de missingness exportee: c'est une preuve methodologique importante
    # pour justifier exclusion, imputation, et limites du memoire.
    missingness_table = (
        cleaned_for_eda.isna().mean()
        .mul(100)
        .round(2)
        .rename("missing_percent")
        .to_frame()
        .assign(missing_count=cleaned_for_eda.isna().sum())
        .sort_values(["missing_percent", "missing_count"], ascending=False)
    )

    # Exclure les colonnes trop manquantes selon le seuil, plus les exclusions
    # forcees documentees dans le memoire/notebooks (TNFa, creatinine, MDRD).
    threshold = float(cfg["cleaning"]["missingness_exclusion_threshold"]) * 100
    threshold_excluded = missingness_table.index[
        missingness_table["missing_percent"] > threshold
    ].tolist()
    forced_excluded = [
        c for c in cfg["cleaning"]["forced_excluded_columns"]
        if c in cleaned_for_eda.columns
    ]
    excluded_columns = sorted(set(threshold_excluded) | set(forced_excluded))

    # Les variables derivees sont retirees avant MissForest pour eviter d'imputer
    # separement des grandeurs calculables. Elles seront recalculees apres.
    derived_columns = [
        c for c in cfg["derived_variables"]["recalculate_after_imputation"]
        if c in cleaned_for_eda.columns
    ]

    pre_imputation = cleaned_for_eda.drop(
        columns=[c for c in excluded_columns + derived_columns if c in cleaned_for_eda.columns]
    )

    # Les categories completes ou imputees par mode sont encodees ensuite pour
    # l'algorithme MissForest, qui travaille sur une matrice numerique.
    categorical_columns = [
        c for c in cfg["schema"]["categorical_columns"]
        if c in pre_imputation.columns
    ]
    profile_col = cfg["schema"]["profile_column"]
    categorical_plus_profile = categorical_columns + (
        [profile_col] if profile_col in pre_imputation.columns else []
    )

    for col in categorical_columns:
        missing_count = pre_imputation[col].isna().sum()
        if missing_count:
            mode = pre_imputation[col].mode(dropna=True)
            if mode.empty:
                raise ValueError(f"Categorical column {col!r} has no observed value for mode imputation.")
            pre_imputation[col] = pre_imputation[col].fillna(mode.iloc[0])

    for col in categorical_plus_profile:
        pre_imputation[col] = pre_imputation[col].astype("category")

    return CleaningResult(
        raw=df,
        cleaned_for_eda=cleaned_for_eda,
        pre_imputation=pre_imputation,
        medication_columns=medication_columns,
        excluded_columns=excluded_columns,
        derived_columns=derived_columns,
        missingness_table=missingness_table,
    )


def recalculate_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Recalcul exact des variables derivees apres imputation des variables sources.
    # Cela maintient les identites mathematiques entre HDL/TG/Totc/LDLc/Glu/Ins/HOMA.
    out = df.copy()
    out["Ratio Tot-c/HDL"] = out["Totc"] / out["HDL"]
    out["Ratio LDL-c/HDL"] = out["LDLc"] / out["HDL"]
    out["Ratio TG/HDL"] = out["TG"] / out["HDL"]
    out["Ratio LogTG/HDL"] = np.log10(out["TG"] / out["HDL"])
    out["Insu pmol/L"] = 6 * out["Ins (µUI/mL)"]
    out["lu(mmol/l)"] = 5.5556 * out["Glu"]
    out["HOMAS%"] = 100 / out["HOMA-IR"]
    return out
