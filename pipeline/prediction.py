from __future__ import annotations

"""Prediction sur un nouveau tableau de patients.

Le module charge les modeles finaux sauvegardes et produit un tableau unique avec
la probabilite de mauvais controle glycemique et la probabilite d'escalade
therapeutique.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import recalculate_derived_variables


def prepare_prediction_features(df: pd.DataFrame, predictors: list[str], cfg: dict) -> pd.DataFrame:
    """Build the same engineered columns used during training, then align predictors."""
    prepared = recalculate_derived_variables(df.copy())

    if "TG" in prepared.columns and "Glu" in prepared.columns and "IMC" in prepared.columns:
        tg_mgdl = pd.to_numeric(prepared["TG"], errors="coerce") * 100
        glu_mgdl = pd.to_numeric(prepared["Glu"], errors="coerce") * 100
        prepared["TyG"] = np.log(tg_mgdl * glu_mgdl / 2)
        prepared["TyG_BMI"] = prepared["TyG"] * pd.to_numeric(prepared["IMC"], errors="coerce")

    friedewald_cfg = cfg.get("feature_engineering", {}).get("friedewald_ldl", {})
    if friedewald_cfg.get("enabled", True) and {"Totc", "HDL", "TG"}.issubset(prepared.columns):
        out_col = friedewald_cfg.get("output_column", "LDLc_Friedewald")
        divisor = float(friedewald_cfg.get("triglyceride_divisor", 5))
        prepared[out_col] = (
            pd.to_numeric(prepared["Totc"], errors="coerce")
            - pd.to_numeric(prepared["HDL"], errors="coerce")
            - (pd.to_numeric(prepared["TG"], errors="coerce") / divisor)
        )

    if "Sexe" in prepared.columns:
        prepared["Sexe_bin"] = (prepared["Sexe"] == "Masculin").astype(int)
    if "HTA" in prepared.columns:
        prepared["HTA_bin"] = (prepared["HTA"] == "HTA").astype(int)
    if "HTS/HTD" in prepared.columns:
        hts_dummies = pd.get_dummies(prepared["HTS/HTD"], prefix="HTS_HTD").astype(int)
        for col in hts_dummies.columns:
            prepared[col] = hts_dummies[col]

    missing = [col for col in predictors if col not in prepared.columns]
    if missing:
        raise ValueError(
            "Prediction dataset is missing required predictor columns after feature engineering: "
            f"{missing}"
        )

    x = prepared[predictors].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if x.isna().any().any():
        missing_counts = x.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            "Prediction dataset contains missing/non-numeric values in model predictors. "
            f"Please impute or complete them first: {missing_counts}"
        )
    return x


def load_model_payload(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def predict_with_model(df: pd.DataFrame, model_path: Path, cfg: dict) -> tuple[np.ndarray, list[str]]:
    payload = load_model_payload(model_path)
    predictors = payload["predictors"]
    x = prepare_prediction_features(df, predictors, cfg)
    proba = payload["model"].predict_proba(x)[:, 1]
    return proba, predictors


def build_dual_prediction_table(
    df: pd.DataFrame,
    cfg: dict,
    models_dir: Path,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    glycemic_target = cfg["target"]["glycemic_control"]["name"]
    escalation_target = cfg["target"]["name"]
    glycemic_model = models_dir / glycemic_target / f"{model_name}_final_model.pkl"
    escalation_model = models_dir / escalation_target / f"{model_name}_final_model.pkl"

    if not glycemic_model.exists():
        raise FileNotFoundError(f"Missing glycemic control model: {glycemic_model}")
    if not escalation_model.exists():
        raise FileNotFoundError(f"Missing escalation model: {escalation_model}")

    glycemic_proba, _ = predict_with_model(df, glycemic_model, cfg)
    escalation_proba, _ = predict_with_model(df, escalation_model, cfg)

    result = df.copy()
    friedewald_cfg = cfg.get("feature_engineering", {}).get("friedewald_ldl", {})
    if friedewald_cfg.get("enabled", True) and {"Totc", "HDL", "TG"}.issubset(result.columns):
        out_col = friedewald_cfg.get("output_column", "LDLc_Friedewald")
        divisor = float(friedewald_cfg.get("triglyceride_divisor", 5))
        result[out_col] = (
            pd.to_numeric(result["Totc"], errors="coerce")
            - pd.to_numeric(result["HDL"], errors="coerce")
            - (pd.to_numeric(result["TG"], errors="coerce") / divisor)
        )
    result["proba_mauvais_controle_glycemique"] = glycemic_proba
    result["prediction_controle_glycemique"] = np.where(
        glycemic_proba >= 0.5,
        cfg["target"]["glycemic_control"]["unfavorable_label"],
        cfg["target"]["glycemic_control"]["favorable_label"],
    )
    result["proba_escalade_therapeutique"] = escalation_proba
    result["prediction_escalade_therapeutique"] = np.where(
        escalation_proba >= 0.5,
        "Escalade_probable",
        "Escalade_peu_probable",
    )
    result["interpretation_synthetique"] = [
        interpret_prediction(g, e) for g, e in zip(glycemic_proba, escalation_proba)
    ]
    return result


def interpret_prediction(glycemic_proba: float, escalation_proba: float) -> str:
    bad_control = glycemic_proba >= 0.5
    escalation = escalation_proba >= 0.5
    if bad_control and escalation:
        return "Profil associe a un controle glycemique defavorable et a une intensification therapeutique probable."
    if bad_control and not escalation:
        return "Profil associe a un controle glycemique defavorable sans intensification therapeutique probable."
    if not bad_control and escalation:
        return "Profil associe a un controle glycemique favorable mais avec intensification therapeutique probable."
    return "Profil associe a un controle glycemique favorable et a une intensification therapeutique peu probable."
