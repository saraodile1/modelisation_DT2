from __future__ import annotations

"""Approche bayesienne guidee par les chemins SEM retenus.

Le module estime des regressions logistiques bayesiennes simples pour les deux
cibles principales du memoire. Les variables sont choisies a partir des blocs
biologiques soutenus par le SEM: anthropometrie, profil immuno-lipidique et
insulinoresistance/statut glycemique observe.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import pickle
import tempfile
import os

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BayesianTargetSpec:
    name: str
    target_column: str
    description: str
    predictors: list[str]
    positive_label: str


@dataclass(frozen=True)
class BayesianFit:
    coefficients: np.ndarray
    covariance: np.ndarray
    predictors: list[str]
    means: pd.Series
    stds: pd.Series
    prior_sds: np.ndarray


def run_bayesian_pipeline(input_path: Path, output_dir: Path, cfg: dict, random_state: int = 42) -> dict[str, object]:
    """Fit SEM-guided Bayesian logistic models and write all artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    target_specs = build_target_specs(df, cfg)
    if not target_specs:
        raise ValueError("No Bayesian target can be built with the available columns.")

    summary_rows = []
    all_coefficients = []
    all_predictions = []
    all_metrics = []

    for spec in target_specs:
        target_dir = output_dir / spec.name
        target_figures_dir = target_dir / "figures"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_figures_dir.mkdir(parents=True, exist_ok=True)

        model_df = df[[spec.target_column, *spec.predictors]].dropna().copy()
        x = model_df[spec.predictors]
        y = model_df[spec.target_column].astype(int)

        holdout = run_holdout_validation(x, y, spec, target_dir, target_figures_dir, random_state)
        cv_summary, cv_rows = run_cross_validation(x, y, spec, target_dir, random_state=random_state)
        full_fit = fit_bayesian_logistic(x, y)
        coefficient_table = summarize_coefficients(full_fit, spec, random_state=random_state)
        patient_predictions = predict_with_uncertainty(full_fit, x, y, spec, random_state=random_state)

        coefficient_table.to_csv(target_dir / "bayesian_coefficients.csv", index=False, encoding="utf-8-sig")
        patient_predictions.to_csv(target_dir / "bayesian_patient_predictions.csv", index=False, encoding="utf-8-sig")
        patient_predictions.to_excel(target_dir / "bayesian_patient_predictions.xlsx", index=False)
        holdout["metrics"].to_csv(target_dir / "bayesian_metrics.csv", index=False, encoding="utf-8-sig")
        cv_rows.to_csv(target_dir / "bayesian_cross_validation_metrics.csv", index=False, encoding="utf-8-sig")

        write_coefficients_plot(coefficient_table, target_figures_dir / "bayesian_coefficients.png")
        write_probability_distribution_plot(
            patient_predictions,
            target_figures_dir / "bayesian_probability_distribution.png",
            title=f"Distribution des probabilites - {spec.name}",
        )
        write_calibration_curve(
            holdout["y_test"],
            holdout["proba_test"],
            target_figures_dir / "bayesian_calibration.png",
            title=f"Calibration bayesienne - {spec.name}",
        )
        write_roc_curve(
            holdout["y_test"],
            holdout["proba_test"],
            holdout["metrics"].iloc[0]["auc"],
            target_figures_dir / "bayesian_roc_curve.png",
            title=f"Courbe ROC bayesienne - {spec.name}",
        )

        model_payload = {
            "fit": full_fit,
            "target": spec,
            "method": "bayesian_logistic_laplace",
            "note": "Regression logistique bayesienne avec prior normal et approximation de Laplace.",
        }
        with (target_dir / "bayesian_model.pkl").open("wb") as f:
            pickle.dump(model_payload, f)

        summary_rows.append(
            {
                "target": spec.name,
                "target_column": spec.target_column,
                "description": spec.description,
                "n": int(len(model_df)),
                "n_predictors": int(len(spec.predictors)),
                "positive_rate": float(y.mean()),
                **{f"holdout_{key}": value for key, value in holdout["metrics"].iloc[0].to_dict().items()},
                **cv_summary,
            }
        )
        all_coefficients.append(coefficient_table.assign(target=spec.name))
        all_predictions.append(patient_predictions.assign(target=spec.name))
        all_metrics.append(holdout["metrics"].assign(target=spec.name))

    summary = pd.DataFrame(summary_rows)
    best_models = select_best_models(summary)
    coefficients_all = pd.concat(all_coefficients, ignore_index=True)
    predictions_all = pd.concat(all_predictions, ignore_index=True)
    metrics_all = pd.concat(all_metrics, ignore_index=True)

    summary.to_csv(output_dir / "bayesian_model_summary.csv", index=False, encoding="utf-8-sig")
    best_models.to_csv(output_dir / "bayesian_best_models.csv", index=False, encoding="utf-8-sig")
    coefficients_all.to_csv(output_dir / "bayesian_coefficients_all_targets.csv", index=False, encoding="utf-8-sig")
    predictions_all.to_csv(output_dir / "bayesian_patient_predictions_all_targets.csv", index=False, encoding="utf-8-sig")
    metrics_all.to_csv(output_dir / "bayesian_metrics_all_targets.csv", index=False, encoding="utf-8-sig")
    write_bayesian_excel_workbook(
        summary,
        best_models,
        metrics_all,
        coefficients_all,
        predictions_all,
        output_dir / "bayesian_results.xlsx",
    )
    write_model_comparison_plot(summary, figures_dir / "bayesian_candidate_model_comparison.png")
    write_summary_report(summary, best_models, coefficients_all, output_dir / "bayesian_summary_report.md")

    return {
        "models": [spec.name for spec in target_specs],
        "output_dir": str(output_dir),
        "n_models": len(target_specs),
    }


def select_best_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Select one candidate per target column using CV AUC then Brier score."""
    rows = []
    for target_column, group in summary.groupby("target_column", sort=False):
        ranked = group.sort_values(["cv_auc_mean", "cv_brier_score_mean"], ascending=[False, True])
        best = ranked.iloc[0].to_dict()
        best["selection_rule"] = "Best CV AUC, with lower CV Brier score as secondary criterion."
        rows.append(best)
    return pd.DataFrame(rows)


def build_target_specs(df: pd.DataFrame, cfg: dict) -> list[BayesianTargetSpec]:
    """Select Bayesian predictors from SEM-guided biological blocks."""
    control_target = cfg["target"]["glycemic_control"]["name"]
    escalation_target = cfg["target"]["name"]
    sem_full_predictors = keep_existing(
        df,
        [
            "IMC",
            "TT",
            "PCSK9",
            "Adipo HMW",
            "HDL",
            "TG",
            "Non HDL",
            "Ins (µUI/mL)",
            "HOMA-IR",
            "HOMAB%",
            "HOMAS%",
        ],
    )
    biomarkers_sources = keep_existing(
        df,
        ["IMC", "TT", "PCSK9", "Adipo HMW", "HDL", "TG", "Non HDL", "Ins (µUI/mL)"],
    )
    homair_block = keep_existing(
        df,
        ["IMC", "TT", "PCSK9", "Adipo HMW", "HDL", "TG", "Non HDL", "HOMA-IR"],
    )
    homab_block = keep_existing(
        df,
        ["IMC", "TT", "PCSK9", "Adipo HMW", "HDL", "TG", "Non HDL", "HOMAB%"],
    )
    parsimonious_control = keep_existing(df, ["HDL", "Ins (µUI/mL)", "HOMAB%", "Adipo HMW"])
    specs: list[BayesianTargetSpec] = []
    if control_target in df.columns:
        control_candidates = [
            (
                "controle_sem_full",
                "Controle glycemique: modele complet guide par les blocs SEM.",
                sem_full_predictors,
            ),
            (
                "controle_biomarkers_sources",
                "Controle glycemique: modele sans indices HOMA derives, centre sur biomarqueurs sources et insuline.",
                biomarkers_sources,
            ),
            (
                "controle_homair_block",
                "Controle glycemique: modele centre sur HOMA-IR et biomarqueurs immuno-lipidiques.",
                homair_block,
            ),
            (
                "controle_homab_block",
                "Controle glycemique: modele centre sur HOMAB% et biomarqueurs immuno-lipidiques.",
                homab_block,
            ),
            (
                "controle_parsimonious_sem",
                "Controle glycemique: modele parcimonieux fonde sur les signaux bayesiens et SEM les plus stables.",
                parsimonious_control,
            ),
        ]
        for name, description, predictors in control_candidates:
            if predictors:
                specs.append(
                    BayesianTargetSpec(
                        name=name,
                        target_column=control_target,
                        description=description,
                        predictors=predictors,
                        positive_label="Profil glycemique defavorable",
                    )
                )
    if escalation_target in df.columns:
        escalation_full = keep_existing(df, [*sem_full_predictors, "Glu"])
        escalation_biomarkers = keep_existing(df, [*biomarkers_sources, "Glu"])
        escalation_lipid = keep_existing(df, ["IMC", "TT", "Glu", "PCSK9", "Adipo HMW", "HDL", "TG", "Non HDL"])
        escalation_nonhdl_focus = keep_existing(df, ["Glu", "Non HDL", "TG", "HDL", "PCSK9", "Adipo HMW", "IMC", "TT"])
        escalation_candidates = [
            (
                "escalade_sem_full",
                "Escalade observee: modele complet guide par les blocs SEM.",
                escalation_full,
            ),
            (
                "escalade_biomarkers_sources",
                "Escalade observee: modele sans indices HOMA derives, centre sur biomarqueurs sources et glycémie.",
                escalation_biomarkers,
            ),
            (
                "escalade_lipid_profile",
                "Escalade observee: modele centre sur le profil lipidique et immuno-lipidique.",
                escalation_lipid,
            ),
            (
                "escalade_nonhdl_focus",
                "Escalade observee: modele parcimonieux centre sur Non-HDL et biomarqueurs proches.",
                escalation_nonhdl_focus,
            ),
        ]
        for name, description, predictors in escalation_candidates:
            if predictors:
                specs.append(
                    BayesianTargetSpec(
                        name=name,
                        target_column=escalation_target,
                        description=description,
                        predictors=predictors,
                        positive_label="Intensite therapeutique observee elevee",
                    )
                )
    return specs


def keep_existing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def fit_bayesian_logistic(x: pd.DataFrame, y: pd.Series) -> BayesianFit:
    """Estimate a Bayesian logistic regression with normal priors.

    The posterior is approximated around the MAP estimate with a Gaussian
    covariance equal to the inverse penalized Hessian.
    """
    from scipy.optimize import minimize

    x_standardized, means, stds = standardize_for_model(x)
    design = np.column_stack([np.ones(len(x_standardized)), x_standardized.to_numpy(dtype=float)])
    y_values = y.to_numpy(dtype=float)
    prior_sds = np.array([2.5, *([1.0] * x_standardized.shape[1])], dtype=float)
    prior_vars = prior_sds**2

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ beta
        proba = sigmoid(linear)
        nll = -np.sum(y_values * np.log(proba + 1e-12) + (1 - y_values) * np.log(1 - proba + 1e-12))
        penalty = 0.5 * np.sum((beta**2) / prior_vars)
        gradient = design.T @ (proba - y_values) + beta / prior_vars
        return float(nll + penalty), gradient

    result = minimize(
        fun=lambda beta: objective(beta)[0],
        jac=lambda beta: objective(beta)[1],
        x0=np.zeros(design.shape[1], dtype=float),
        method="BFGS",
    )
    beta_map = result.x
    proba = sigmoid(design @ beta_map)
    weights = proba * (1 - proba)
    hessian = design.T @ (weights[:, None] * design) + np.diag(1 / prior_vars)
    covariance = np.linalg.pinv(hessian)
    return BayesianFit(
        coefficients=beta_map,
        covariance=covariance,
        predictors=list(x.columns),
        means=means,
        stds=stds,
        prior_sds=prior_sds,
    )


def standardize_for_model(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    numeric = x.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean()
    stds = numeric.std(ddof=0).replace(0, 1.0)
    return (numeric - means).divide(stds).fillna(0.0), means, stds


def standardize_with_fit(x: pd.DataFrame, fit: BayesianFit) -> pd.DataFrame:
    numeric = x[fit.predictors].apply(pd.to_numeric, errors="coerce")
    return (numeric - fit.means).divide(fit.stds).fillna(0.0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -35, 35)))


def posterior_samples(fit: BayesianFit, random_state: int, n_samples: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.multivariate_normal(fit.coefficients, nearest_positive_definite(fit.covariance), size=n_samples)


def summarize_coefficients(fit: BayesianFit, spec: BayesianTargetSpec, random_state: int) -> pd.DataFrame:
    samples = posterior_samples(fit, random_state=random_state)
    names = ["Intercept", *fit.predictors]
    rows = []
    for idx, name in enumerate(names):
        draws = samples[:, idx]
        rows.append(
            {
                "target": spec.name,
                "predictor": name,
                "posterior_mean": float(draws.mean()),
                "posterior_sd": float(draws.std(ddof=0)),
                "credible_low_95": float(np.quantile(draws, 0.025)),
                "credible_high_95": float(np.quantile(draws, 0.975)),
                "probability_positive": float(np.mean(draws > 0)),
                "credible_excludes_zero": bool(
                    np.quantile(draws, 0.025) > 0 or np.quantile(draws, 0.975) < 0
                ),
                "prior_sd": float(fit.prior_sds[idx]),
                "interpretation": coefficient_interpretation(draws, name),
            }
        )
    return pd.DataFrame(rows)


def coefficient_interpretation(draws: np.ndarray, name: str) -> str:
    low = np.quantile(draws, 0.025)
    high = np.quantile(draws, 0.975)
    mean = draws.mean()
    if name == "Intercept":
        return "Niveau moyen de probabilite avant contribution des predicteurs standardises."
    if low > 0:
        return "Association probabilistiquement positive avec la cible."
    if high < 0:
        return "Association probabilistiquement negative avec la cible."
    direction = "positive" if mean > 0 else "negative"
    return f"Direction moyenne {direction}, mais incertitude compatible avec une absence d'association nette."


def predict_with_uncertainty(
    fit: BayesianFit,
    x: pd.DataFrame,
    y: pd.Series,
    spec: BayesianTargetSpec,
    random_state: int,
) -> pd.DataFrame:
    x_standardized = standardize_with_fit(x, fit)
    design = np.column_stack([np.ones(len(x_standardized)), x_standardized.to_numpy(dtype=float)])
    samples = posterior_samples(fit, random_state=random_state + 10)
    proba_draws = sigmoid(design @ samples.T)
    mean = proba_draws.mean(axis=1)
    low = np.quantile(proba_draws, 0.025, axis=1)
    high = np.quantile(proba_draws, 0.975, axis=1)
    width = high - low
    return pd.DataFrame(
        {
            "patient_index": np.arange(len(x), dtype=int),
            "target": spec.name,
            "observed_class": y.to_numpy(dtype=int),
            "probability_mean": mean,
            "credible_low_95": low,
            "credible_high_95": high,
            "credible_interval_width": width,
            "predicted_class": (mean >= 0.5).astype(int),
            "uncertainty_level": [uncertainty_label(value) for value in width],
            "positive_label": spec.positive_label,
        }
    )


def uncertainty_label(width: float) -> str:
    if width < 0.15:
        return "faible"
    if width < 0.30:
        return "moderee"
    return "elevee"


def run_holdout_validation(
    x: pd.DataFrame,
    y: pd.Series,
    spec: BayesianTargetSpec,
    output_dir: Path,
    figures_dir: Path,
    random_state: int,
) -> dict[str, object]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    fit = fit_bayesian_logistic(x.iloc[idx_train], y.iloc[idx_train])
    predictions = predict_with_uncertainty(
        fit,
        x.iloc[idx_test].reset_index(drop=True),
        y.iloc[idx_test].reset_index(drop=True),
        spec,
        random_state=random_state + 100,
    )
    proba_test = predictions["probability_mean"].to_numpy(dtype=float)
    y_test = y.iloc[idx_test].reset_index(drop=True)
    metrics = pd.DataFrame([compute_binary_metrics(y_test, proba_test)])
    predictions.to_csv(output_dir / "bayesian_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    write_confusion_matrix_plot(
        y_test,
        (proba_test >= 0.5).astype(int),
        figures_dir / "bayesian_confusion_matrix.png",
        title=f"Matrice de confusion bayesienne - {spec.name}",
    )
    return {"metrics": metrics, "y_test": y_test, "proba_test": proba_test}


def run_cross_validation(
    x: pd.DataFrame,
    y: pd.Series,
    spec: BayesianTargetSpec,
    output_dir: Path,
    random_state: int,
    n_splits: int = 5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compare Bayesian candidate models with stratified cross-validation."""
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (idx_train, idx_test) in enumerate(splitter.split(x, y), start=1):
        fit = fit_bayesian_logistic(x.iloc[idx_train], y.iloc[idx_train])
        proba = predict_map_probability(fit, x.iloc[idx_test])
        metrics = compute_binary_metrics(y.iloc[idx_test].reset_index(drop=True), proba)
        rows.append(
            {
                "model": spec.name,
                "target_column": spec.target_column,
                "fold": fold,
                "n_train": int(len(idx_train)),
                "n_test": int(len(idx_test)),
                **metrics,
            }
        )

    cv = pd.DataFrame(rows)
    summary = {
        "cv_n_splits": int(n_splits),
        "cv_auc_mean": float(cv["auc"].mean()),
        "cv_auc_std": float(cv["auc"].std(ddof=0)),
        "cv_auc_pr_mean": float(cv["auc_pr"].mean()),
        "cv_auc_pr_std": float(cv["auc_pr"].std(ddof=0)),
        "cv_accuracy_mean": float(cv["accuracy"].mean()),
        "cv_accuracy_std": float(cv["accuracy"].std(ddof=0)),
        "cv_balanced_accuracy_mean": float(cv["balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(cv["balanced_accuracy"].std(ddof=0)),
        "cv_sensitivity_mean": float(cv["sensitivity"].mean()),
        "cv_sensitivity_std": float(cv["sensitivity"].std(ddof=0)),
        "cv_specificity_mean": float(cv["specificity"].mean()),
        "cv_specificity_std": float(cv["specificity"].std(ddof=0)),
        "cv_precision_mean": float(cv["precision"].mean()),
        "cv_precision_std": float(cv["precision"].std(ddof=0)),
        "cv_f1_mean": float(cv["f1"].mean()),
        "cv_f1_std": float(cv["f1"].std(ddof=0)),
        "cv_brier_score_mean": float(cv["brier_score"].mean()),
        "cv_brier_score_std": float(cv["brier_score"].std(ddof=0)),
    }
    return summary, cv


def predict_map_probability(fit: BayesianFit, x: pd.DataFrame) -> np.ndarray:
    x_standardized = standardize_with_fit(x, fit)
    design = np.column_stack([np.ones(len(x_standardized)), x_standardized.to_numpy(dtype=float)])
    return sigmoid(design @ fit.coefficients)


def compute_binary_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "auc_pr": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    eigvals = np.maximum(eigvals, 1e-9)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def write_bayesian_excel_workbook(
    summary: pd.DataFrame,
    best_models: pd.DataFrame,
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    predictions: pd.DataFrame,
    path: Path,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        best_models.to_excel(writer, sheet_name="best_models", index=False)
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        coefficients.to_excel(writer, sheet_name="coefficients", index=False)
        predictions.to_excel(writer, sheet_name="patient_predictions", index=False)


def write_summary_report(
    summary: pd.DataFrame,
    best_models: pd.DataFrame,
    coefficients: pd.DataFrame,
    path: Path,
) -> None:
    strong = coefficients[
        (coefficients["predictor"] != "Intercept") & (coefficients["credible_excludes_zero"])
    ].copy()
    lines = [
        "# Resume bayesien",
        "",
        "Les modeles bayesiens estiment des probabilites individualisees avec intervalles credibles.",
        "Ils sont guides par les blocs biologiques retenus apres SEM, sans chercher a demontrer une evolution longitudinale.",
        "",
        "## Performance holdout",
        "",
        markdown_table(summary.round(4)),
        "",
        "## Meilleurs modeles par cible",
        "",
        markdown_table(best_models.round(4)),
        "",
        "## Coefficients dont l'intervalle credible exclut zero",
        "",
        markdown_table(
            strong[
                [
                    "target",
                    "predictor",
                    "posterior_mean",
                    "credible_low_95",
                    "credible_high_95",
                    "probability_positive",
                    "interpretation",
                ]
            ].round(4)
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Aucune ligne._"
    values = df.map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(values.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, sep, *rows])


def _require_matplotlib():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_coefficients_plot(coefficients: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    plot_df = coefficients[coefficients["predictor"] != "Intercept"].copy()
    plot_df = plot_df.sort_values("posterior_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    xerr = np.vstack(
        [
            plot_df["posterior_mean"] - plot_df["credible_low_95"],
            plot_df["credible_high_95"] - plot_df["posterior_mean"],
        ]
    )
    colors = np.where(plot_df["credible_excludes_zero"], "#2f6db3", "#9aa5b1")
    ax.errorbar(plot_df["posterior_mean"], plot_df["predictor"], xerr=xerr, fmt="o", color="#333333", ecolor=colors)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Coefficient posterior moyen et intervalle credible 95%")
    ax.set_title("Coefficients bayesiens")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_probability_distribution_plot(predictions: pd.DataFrame, path: Path, title: str) -> None:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(predictions["probability_mean"], bins=18, color="#54a24b", edgecolor="white")
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Probabilite posterior moyenne")
    ax.set_ylabel("Nombre de patients")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_model_comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    plot_df = summary.sort_values(["target_column", "cv_auc_mean"], ascending=[True, False]).copy()
    labels = [f"{row.target_column}\n{row.target}" for row in plot_df.itertuples(index=False)]
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, plot_df["cv_auc_mean"], color="#2f6db3", label="AUC CV")
    ax.scatter(x, 1 - plot_df["cv_brier_score_mean"], color="#f58518", label="1 - Brier CV", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des modeles bayesiens candidats")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_calibration_curve(y_true: pd.Series, proba: np.ndarray, path: Path, title: str) -> None:
    from sklearn.calibration import calibration_curve

    plt = _require_matplotlib()
    fraction_positive, mean_predicted = calibration_curve(y_true, proba, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_predicted, fraction_positive, marker="o", color="#54a24b", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("Probabilite estimee")
    ax.set_ylabel("Frequence observee")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_roc_curve(y_true: pd.Series, proba: np.ndarray, auc_value: float, path: Path, title: str) -> None:
    from sklearn.metrics import roc_curve

    plt = _require_matplotlib()
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}", color="#2f6db3", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_confusion_matrix_plot(y_true: pd.Series, y_pred: np.ndarray, path: Path, title: str) -> None:
    from sklearn.metrics import confusion_matrix

    plt = _require_matplotlib()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Reel 0", "Reel 1"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Classe observee")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
