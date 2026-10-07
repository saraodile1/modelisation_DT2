from __future__ import annotations

"""Feature engineering et construction des cibles ML.

Deux cibles sont construites:
- Controle_glycemique_binaire, cible principale derivee du profil glycemique;
- Escalade_therapeutique, cible secondaire d'intensification du traitement.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import infer_medication_columns


def build_escalation_target(raw: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    # La cible est construite depuis les donnees brutes structurees, jamais depuis
    # les doses imputees, afin d'eviter qu'une imputation influence y.
    antidiabetics = cfg["schema"]["antidiabetic_columns"]
    missing = [c for c in antidiabetics if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing antidiabetic columns required for target construction: {missing}")

    # Regle actuelle: dose differente de 0 = medicament pris. Cela inclut la
    # sentinelle configurable (-1 dans ton dataset), qui signifie dose inconnue
    # mais medicament effectivement pris.
    n_antidiabetics = sum((raw[col] != 0).astype(int) for col in antidiabetics)
    target_cfg = cfg["target"]
    labels = {
        0: target_cfg["none_label"],
        1: target_cfg["mono_label"],
        2: target_cfg["bi_label"],
        3: target_cfg["tri_label"],
    }
    # Variable descriptive a 4 niveaux, puis cible binaire pour le ML.
    # La classe 0 regroupe "aucun traitement" et "monotherapie": absence
    # d'intensification observee. La classe 1 regroupe bitherapie/tritherapie.
    therapeutic_level = n_antidiabetics.map(labels)
    escalation = n_antidiabetics.map({0: 0, 1: 0, 2: 1, 3: 1})
    return therapeutic_level, escalation


def build_glycemic_control_target(raw: pd.DataFrame, cfg: dict) -> pd.Series:
    # Cible principale du memoire avec les donnees disponibles: elle indique si
    # le profil glycemique observe est favorable ou defavorable.
    profile_col = cfg["schema"]["profile_column"]
    target_cfg = cfg["target"]["glycemic_control"]
    if profile_col not in raw.columns:
        raise ValueError(f"Missing profile column required for glycemic target: {profile_col!r}")

    favorable = set(target_cfg["favorable_profiles"])
    unfavorable = set(target_cfg["unfavorable_profiles"])
    target = pd.Series(np.nan, index=raw.index, name=target_cfg["name"])
    target.loc[raw[profile_col].isin(favorable)] = 0
    target.loc[raw[profile_col].isin(unfavorable)] = 1
    return target


def add_feature_engineering(
    datasets: list[pd.DataFrame],
    raw: pd.DataFrame,
    cfg: dict,
) -> dict[str, object]:
    # Copie defensive: on ne modifie pas les datasets imputes source.
    therapeutic_level, escalation = build_escalation_target(raw, cfg)
    glycemic_control = build_glycemic_control_target(raw, cfg)
    engineered = [d.copy() for d in datasets]

    for d in engineered:
        d[cfg["target"]["descriptive_name"]] = therapeutic_level.to_numpy()
        d[cfg["target"]["name"]] = escalation.to_numpy()
        d[cfg["target"]["glycemic_control"]["name"]] = glycemic_control.to_numpy()

        # TyG-BMI est ajoute apres imputation, a partir de TG, Glu et IMC.
        tg_mgdl = d["TG"] * 100
        glu_mgdl = d["Glu"] * 100
        d["TyG"] = np.log(tg_mgdl * glu_mgdl / 2)
        d["TyG_BMI"] = d["TyG"] * d["IMC"]

        # Formule de Friedewald: estimation du LDL-C a partir du bilan lipidique.
        # Les donnees actuelles sont coherentes avec LDL-C = Totc - HDL - TG/5.
        friedewald_cfg = cfg.get("feature_engineering", {}).get("friedewald_ldl", {})
        if friedewald_cfg.get("enabled", True):
            out_col = friedewald_cfg.get("output_column", "LDLc_Friedewald")
            divisor = float(friedewald_cfg.get("triglyceride_divisor", 5))
            d[out_col] = d["Totc"] - d["HDL"] - (d["TG"] / divisor)

        # Encodage simple pour rendre les categories utilisables par scikit-learn.
        d["Sexe_bin"] = (d["Sexe"] == "Masculin").astype(int)
        d["HTA_bin"] = (d["HTA"] == "HTA").astype(int)
        hts_dummies = pd.get_dummies(d["HTS/HTD"], prefix="HTS_HTD").astype(int)
        for col in hts_dummies.columns:
            d[col] = hts_dummies[col]

    # Exclure les medicaments des predicteurs pour eviter de predire l'intensite
    # du traitement a partir des colonnes qui decrivent directement les traitements.
    medication_columns = infer_medication_columns(raw, cfg)
    excluded_predictors = set(medication_columns)
    excluded_predictors.update(cfg["derived_variables"]["redundant_predictors_to_exclude"])
    target_columns = [
        cfg["target"]["descriptive_name"],
        cfg["target"]["name"],
        cfg["target"]["glycemic_control"]["name"],
    ]
    excluded_predictors.update([cfg["schema"]["profile_column"], "Sexe", "HTA", "HTS/HTD", "TyG"])
    excluded_predictors.update(target_columns)

    base_predictors = [c for c in engineered[0].columns if c not in excluded_predictors]
    targets = {
        cfg["target"]["glycemic_control"]["name"]: build_target_dataset(
            engineered,
            target_name=cfg["target"]["glycemic_control"]["name"],
            predictors=[
                c for c in base_predictors
                if c not in set(cfg["target"]["glycemic_control"].get("excluded_predictors", []))
            ],
            target_kind="controle_glycemique",
        ),
        cfg["target"]["name"]: build_target_dataset(
            engineered,
            target_name=cfg["target"]["name"],
            predictors=base_predictors,
            target_kind="escalade_therapeutique",
        ),
    }

    return {
        "datasets": engineered,
        "targets": targets,
        # Compatibilite: l'ancien format pointe vers l'analyse secondaire.
        "X_list": targets[cfg["target"]["name"]]["X_list"],
        "y_list": targets[cfg["target"]["name"]]["y_list"],
        "predicteurs": targets[cfg["target"]["name"]]["predicteurs"],
        "target_distribution": targets[cfg["target"]["name"]]["target_distribution"],
    }


def build_target_dataset(
    engineered: list[pd.DataFrame],
    target_name: str,
    predictors: list[str],
    target_kind: str,
) -> dict[str, object]:
    # Une paire X/y par dataset impute. y est identique, X varie avec l'imputation.
    x_list: list[pd.DataFrame] = []
    y_list: list[pd.Series] = []
    for d in engineered:
        mask = d[target_name].notna()
        x_list.append(d.loc[mask, predictors].reset_index(drop=True))
        y_list.append(d.loc[mask, target_name].astype(int).reset_index(drop=True))

    return {
        "target_name": target_name,
        "target_kind": target_kind,
        "X_list": x_list,
        "y_list": y_list,
        "predicteurs": predictors,
        "target_distribution": engineered[0][target_name].value_counts(dropna=False).to_dict(),
    }


def save_features(feature_data: dict[str, object], output_dir: Path) -> None:
    # Export compact pour la modelisation et listes lisibles des predicteurs retenus.
    output_dir.mkdir(parents=True, exist_ok=True)
    engineered = feature_data["datasets"][0]  # type: ignore[index]
    engineered.to_excel(output_dir / "engineered_dataset.xlsx", index=False)
    with (output_dir / "engineered_dataset.pkl").open("wb") as f:
        pickle.dump(engineered, f)

    targets: dict[str, dict[str, object]] = feature_data["targets"]  # type: ignore[assignment]
    summary_rows = []
    for target_name, target_data in targets.items():
        target_dir = output_dir / str(target_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        with (target_dir / "X_y_datasets_imputes.pkl").open("wb") as f:
            pickle.dump(
                {
                    "X_list": target_data["X_list"],
                    "y_list": target_data["y_list"],
                    "predicteurs": target_data["predicteurs"],
                },
                f,
            )
        pd.Series(target_data["predicteurs"], name="predictor").to_csv(
            target_dir / "predictors.csv", index=False, encoding="utf-8-sig"
        )
        for value, count in target_data["target_distribution"].items():  # type: ignore[union-attr]
            summary_rows.append(
                {
                    "target": target_name,
                    "class_value": value,
                    "count": count,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "targets_distribution.csv", index=False, encoding="utf-8-sig"
    )
