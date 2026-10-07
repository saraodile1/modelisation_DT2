from __future__ import annotations

"""Point d'entree du pipeline scientifique.

Ce script part du dataset deja structure (une ligne par patient, colonnes biologiques,
colonnes medicamenteuses, Profil_glycemique). Il ne refait pas toute la lecture du
fichier brut: cette partie reste dans le dossier Nettoyage/.

Les options permettent de verifier progressivement le pipeline:
- stop apres nettoyage/EDA;
- reutiliser les jeux imputes existants;
- entrainer ou non le modele.
"""

import argparse
import pickle

from pipeline.cleaning import prepare_cleaning
from pipeline.config import ensure_output_tree, load_config
from pipeline.eda import build_eda_tables, write_eda_figures, write_eda_tables
from pipeline.features import add_feature_engineering, save_features
from pipeline.imputation import run_imputation
from pipeline.io import read_table, write_json, write_text
from pipeline.modeling import run_random_forest
from pipeline.schema import validate_structured_dataset


def run(
    config_path: str,
    skip_imputation: bool = False,
    skip_modeling: bool = False,
    stop_after_pre_imputation: bool = False,
) -> None:
    # 1) Charger la configuration: toutes les hypotheses modifiables doivent etre
    # dans le YAML, pas cachees dans le code Python.
    config = load_config(config_path)
    paths = ensure_output_tree(config)
    cfg = config.raw

    # 2) Lire le dataset structure et verifier qu'il contient les colonnes minimales
    # necessaires au memoire: biomarqueurs principaux, traitements antidiabetiques,
    # et profil glycemique.
    df = read_table(config.input_dataset)
    schema_report = validate_structured_dataset(df, cfg)
    write_json(
        {
            "missing_required": schema_report.missing_required,
            "medication_columns": schema_report.medication_columns,
            "additional_medication_columns": schema_report.additional_medication_columns,
        },
        paths["reports"] / "schema_report.json",
    )
    if not schema_report.is_valid:
        raise ValueError(
            "Input dataset does not match the required structured schema. "
            f"Missing columns: {schema_report.missing_required}"
        )

    # 3) Nettoyage analytique: cette etape correspond aux corrections decouvertes
    # pendant l'EDA (sentinelles, zeros impossibles, HOMA en texte, colonnes trop
    # manquantes, variables derivees mises de cote avant imputation).
    cleaning = prepare_cleaning(df, cfg)
    cleaning.cleaned_for_eda.to_excel(paths["tables"] / "dataset_cleaned_for_eda.xlsx", index=False)
    cleaning.pre_imputation.to_excel(paths["tables"] / "dataset_pre_imputation.xlsx", index=False)
    cleaning.missingness_table.to_csv(
        paths["tables"] / "missingness.csv", encoding="utf-8-sig", index_label="variable"
    )

    # 4) EDA reproductible: on exporte des tables CSV pour documenter la qualite
    # des donnees, sans dependre d'un notebook ouvert manuellement.
    eda_tables = build_eda_tables(df, cleaning.cleaned_for_eda, cfg)
    write_eda_tables(eda_tables, paths["tables"])
    write_eda_figures(eda_tables, paths["figures"] / "eda")

    # Mode diagnostic: utile quand une autre personne arrive avec son dataset.
    # On peut verifier schema + nettoyage + missingness avant de lancer l'imputation.
    if stop_after_pre_imputation:
        write_text(
            build_run_summary(cfg, cleaning, {"target_distribution": {}, "predicteurs": []}, {}),
            paths["reports"] / "run_summary.md",
        )
        return

    # 5) Imputation configuree. Pour gagner du temps pendant le developpement, on peut
    # reutiliser une sortie imputee existante. Sans --skip-imputation, la methode
    # indiquee dans config/pipeline.yaml est relancee depuis le dataset pre-imputation.
    if skip_imputation:
        existing_imputed = cfg["paths"].get("existing_imputed_datasets")
        imputed_path = config.root / existing_imputed if existing_imputed else paths["imputed"] / "missforest_dataset.pkl"
        with imputed_path.open("rb") as f:
            datasets = pickle.load(f)
    elif cfg["imputation"].get("enabled", True):
        datasets = run_imputation(cleaning, cfg, paths["imputed"], config.random_state)
    else:
        raise RuntimeError("Imputation is disabled and no imputed datasets were loaded.")

    # 6) Feature engineering: construction des deux cibles ML
    # Controle_glycemique_binaire et Escalade_therapeutique, encodage des
    # categories, TyG-BMI, puis matrices X/y separees par cible.
    feature_data = add_feature_engineering(datasets, cleaning.raw, cfg)
    save_features(feature_data, paths["features"])

    # 7) Modelisation: validation croisee stratifiee puis controle secondaire par
    # split train/test simple, avec Random Forest et XGBoost pour chaque cible.
    metrics = {}
    if cfg["modeling"].get("enabled", True) and not skip_modeling:
        metrics = run_random_forest(feature_data, cfg, paths["models"], config.random_state)
        write_json(metrics, paths["models"] / "pooled_metrics.json")

    write_text(build_run_summary(cfg, cleaning, feature_data, metrics), paths["reports"] / "run_summary.md")


def build_run_summary(cfg, cleaning, feature_data, metrics: dict) -> str:
    # Rapport court et humainement lisible: il sert a tracer ce qui a ete lance,
    # les dimensions des donnees, la cible, et les performances principales.
    targets = feature_data.get("targets", {})
    target_distribution = feature_data.get("target_distribution", {})
    predictors = feature_data.get("predicteurs", [])
    lines = [
        "# Pipeline run summary",
        "",
        "## Scope",
        "",
        "This pipeline starts from the structured dataset produced after the initial Nettoyage notebooks.",
        "Each run applies schema validation, analytical cleaning, EDA tables and pre-imputation preparation.",
        "Depending on the selected options, it then runs or reuses the configured imputation method, builds ML features and trains the model.",
        "",
        "## Cleaning",
        "",
        f"- Raw shape: {cleaning.raw.shape[0]} rows x {cleaning.raw.shape[1]} columns",
        f"- Pre-imputation shape: {cleaning.pre_imputation.shape[0]} rows x {cleaning.pre_imputation.shape[1]} columns",
        f"- Medication columns inferred: {len(cleaning.medication_columns)}",
        f"- Excluded columns: {', '.join(cleaning.excluded_columns)}",
        f"- Derived columns recalculated after imputation: {', '.join(cleaning.derived_columns)}",
        "",
        "## Target",
        "",
        f"- Primary target: {cfg['target']['glycemic_control']['name']}",
        f"- Secondary target: {cfg['target']['name']}",
        "- Important: glycemic control is an observed cross-sectional status, while therapeutic escalation is an intensity-of-care proxy.",
    ]
    if targets:
        for target_name, target_data in targets.items():
            lines.extend(
                [
                    f"- {target_name} distribution: {target_data.get('target_distribution')}",
                    f"- {target_name} predictors retained: {len(target_data.get('predicteurs', []))}",
                ]
            )
    else:
        lines.extend(
            [
                f"- Target distribution: {target_distribution or 'not computed in this short run'}",
                f"- Predictors retained: {len(predictors)}",
            ]
        )
    if metrics and "targets" in metrics:
        lines.extend(["", "## Modeling", ""])
        for target_name, target_metrics in metrics["targets"].items():
            lines.append(f"### {target_name}")
            lines.append(f"- Cross-validation folds: {target_metrics.get('cv_n_splits', 'not computed')}")
            lines.append(f"- Best model by CV AUC: {target_metrics.get('best_model_cv_auc', 'not computed')}")
            for model_name, model_metrics in target_metrics["models"].items():
                lines.extend(
                    [
                        f"- {model_name} CV mean AUC: {model_metrics['cv_auc_mean']:.3f}",
                        f"- {model_name} CV mean AUC PR: {model_metrics['cv_auc_pr_mean']:.3f}",
                        f"- {model_name} CV mean accuracy: {model_metrics['cv_accuracy_mean']:.3f}",
                        f"- {model_name} CV mean balanced accuracy: {model_metrics['cv_balanced_accuracy_mean']:.3f}",
                        f"- {model_name} CV mean sensitivity: {model_metrics['cv_sensitivity_mean']:.3f}",
                        f"- {model_name} CV mean specificity: {model_metrics['cv_specificity_mean']:.3f}",
                        f"- {model_name} CV mean precision: {model_metrics['cv_precision_mean']:.3f}",
                        f"- {model_name} CV mean F1-score: {model_metrics['cv_f1_mean']:.3f}",
                        f"- {model_name} CV mean Brier score: {model_metrics['cv_brier_score_mean']:.3f}",
                        f"- {model_name} holdout AUC: {model_metrics['auc_pooled']:.3f}",
                        f"- {model_name} holdout accuracy: {model_metrics['accuracy_pooled']:.3f}",
                        f"- {model_name} holdout sensitivity: {model_metrics['sensitivity_pooled']:.3f}",
                        f"- {model_name} holdout specificity: {model_metrics['specificity_pooled']:.3f}",
                    ]
                )
    elif metrics and "models" in metrics:
        lines.extend(["", "## Modeling", ""])
        lines.append(f"- Cross-validation folds: {metrics.get('cv_n_splits', 'not computed')}")
        lines.append(f"- Best model by CV AUC: {metrics.get('best_model_cv_auc', 'not computed')}")
        for model_name, model_metrics in metrics["models"].items():
            lines.extend(
                [
                    f"- {model_name} CV mean AUC: {model_metrics['cv_auc_mean']:.3f}",
                    f"- {model_name} CV mean AUC PR: {model_metrics['cv_auc_pr_mean']:.3f}",
                    f"- {model_name} CV mean accuracy: {model_metrics['cv_accuracy_mean']:.3f}",
                    f"- {model_name} CV mean balanced accuracy: {model_metrics['cv_balanced_accuracy_mean']:.3f}",
                    f"- {model_name} CV mean sensitivity: {model_metrics['cv_sensitivity_mean']:.3f}",
                    f"- {model_name} CV mean specificity: {model_metrics['cv_specificity_mean']:.3f}",
                    f"- {model_name} CV mean precision: {model_metrics['cv_precision_mean']:.3f}",
                    f"- {model_name} CV mean F1-score: {model_metrics['cv_f1_mean']:.3f}",
                    f"- {model_name} CV mean Brier score: {model_metrics['cv_brier_score_mean']:.3f}",
                    f"- {model_name} holdout AUC: {model_metrics['auc_pooled']:.3f}",
                    f"- {model_name} holdout accuracy: {model_metrics['accuracy_pooled']:.3f}",
                    f"- {model_name} holdout sensitivity: {model_metrics['sensitivity_pooled']:.3f}",
                    f"- {model_name} holdout specificity: {model_metrics['specificity_pooled']:.3f}",
                ]
            )
    elif metrics:
        lines.extend(
            [
                "",
                "## Modeling",
                "",
                f"- Cross-validation folds: {metrics.get('cv_n_splits', 'not computed')}",
                f"- CV mean AUC: {metrics.get('cv_auc_mean', 0):.3f}",
                f"- CV mean accuracy: {metrics.get('cv_accuracy_mean', 0):.3f}",
                f"- Pooled AUC: {metrics['auc_pooled']:.3f}",
                f"- Pooled accuracy: {metrics['accuracy_pooled']:.3f}",
                f"- Mean individual AUC across imputed datasets: {metrics['auc_mean_individual']:.3f}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    # Interface en ligne de commande. Les options sont volontairement simples pour
    # qu'une autre personne puisse relancer le pipeline sans ouvrir le code.
    parser = argparse.ArgumentParser(description="Run the DT2 thesis practical pipeline.")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline YAML config.")
    parser.add_argument("--skip-imputation", action="store_true", help="Reuse the imputed dataset configured in paths.existing_imputed_datasets.")
    parser.add_argument("--skip-modeling", action="store_true", help="Stop after feature engineering.")
    parser.add_argument(
        "--stop-after-pre-imputation",
        action="store_true",
        help="Validate schema, analytical cleaning and EDA tables, then stop before imputation.",
    )
    args = parser.parse_args()
    run(
        args.config,
        skip_imputation=args.skip_imputation,
        skip_modeling=args.skip_modeling,
        stop_after_pre_imputation=args.stop_after_pre_imputation,
    )


if __name__ == "__main__":
    main()
