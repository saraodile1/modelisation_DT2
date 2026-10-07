from __future__ import annotations

"""SEM separe du pipeline ML, sous forme de path analysis observee.

Le module teste des chemins definis a l'avance entre sources cliniques,
biomarqueurs et statuts metaboliques observes. Il ne cherche pas a decouvrir
automatiquement un graphe: les modeles sont volontairement explicites pour
rester alignes avec le memoire et le caractere transversal de la base.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EquationSpec:
    # Chaque equation represente un chemin SEM local: variable dependante ~ predicteurs.
    outcome: str
    predictors: list[str]


@dataclass(frozen=True)
class SemModelSpec:
    name: str
    description: str
    sources: list[str]
    covariates: list[str]
    mediators: list[str]
    outcome: str
    equations: list[EquationSpec]


def run_sem_pipeline(input_path: Path, output_dir: Path, random_state: int = 42) -> dict[str, object]:
    """Prepare SEM data, fit configured path models and write all artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    models = build_sem_model_specs(df)
    sem_columns = sorted({col for model in models for col in model_columns(model)})
    sem_df = prepare_sem_dataset(df, sem_columns)
    sem_df.to_excel(output_dir / "sem_dataset.xlsx", index=False)
    sem_df.to_pickle(output_dir / "sem_dataset.pkl")
    write_sem_correlation_heatmap(sem_df, figures_dir / "sem_correlation_heatmap.png")
    write_smartpls_conceptual_diagram(figures_dir / "sem_smartpls_conceptual_model.png")

    summary_rows = []
    all_parameters = []
    all_effects = []
    all_fit = []
    all_global_fit = []

    for model in models:
        model_dir = output_dir / model.name
        model_figures_dir = model_dir / "figures"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_figures_dir.mkdir(parents=True, exist_ok=True)

        model_df = sem_df[model_columns(model)].dropna().copy()
        standardized = standardize(model_df)
        parameters, fit = fit_path_model(standardized, model)
        global_fit = compute_global_fit_indices(standardized, model, parameters, fit)
        effects = compute_effects(parameters, model)
        global_fit_table = pd.DataFrame([global_fit])

        parameters.to_csv(model_dir / "sem_parameters.csv", index=False, encoding="utf-8-sig")
        fit.to_csv(model_dir / "sem_fit_metrics.csv", index=False, encoding="utf-8-sig")
        global_fit_table.to_csv(model_dir / "sem_global_fit_metrics.csv", index=False, encoding="utf-8-sig")
        effects.to_csv(model_dir / "sem_effects.csv", index=False, encoding="utf-8-sig")
        write_model_spec(model, model_dir / "sem_model_spec.txt")
        write_path_diagram(model, parameters, model_figures_dir / "sem_paths.png")
        write_smartpls_model_diagram(model, parameters, model_figures_dir / "sem_smartpls_model.png")
        write_coefficients_plot(parameters, model_figures_dir / "sem_standardized_coefficients.png")
        write_effects_plot(effects, model_figures_dir / "sem_direct_indirect_total_effects.png")
        write_model_report(model, parameters, fit, global_fit_table, effects, model_dir / "sem_report.md")

        all_parameters.append(parameters.assign(model=model.name))
        all_effects.append(effects.assign(model=model.name))
        all_fit.append(fit.assign(model=model.name))
        all_global_fit.append(global_fit_table.assign(model=model.name))
        summary_rows.append(build_summary_row(model, parameters, fit, global_fit, effects, len(model_df)))

    parameters_all = pd.concat(all_parameters, ignore_index=True)
    effects_all = pd.concat(all_effects, ignore_index=True)
    fit_all = pd.concat(all_fit, ignore_index=True)
    global_fit_all = pd.concat(all_global_fit, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    parameters_all.to_csv(output_dir / "sem_parameters_all_models.csv", index=False, encoding="utf-8-sig")
    effects_all.to_csv(output_dir / "sem_effects_all_models.csv", index=False, encoding="utf-8-sig")
    fit_all.to_csv(output_dir / "sem_fit_metrics_all_models.csv", index=False, encoding="utf-8-sig")
    global_fit_all.to_csv(output_dir / "sem_global_fit_metrics_all_models.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "sem_model_summary.csv", index=False, encoding="utf-8-sig")
    write_sem_excel_workbook(summary, fit_all, global_fit_all, parameters_all, effects_all, output_dir / "sem_results.xlsx")
    write_sem_summary_report(summary, output_dir / "sem_summary_report.md")

    return {
        "n_rows": int(len(sem_df)),
        "models": [model.name for model in models],
        "output_dir": str(output_dir),
    }


def build_sem_model_specs(df: pd.DataFrame) -> list[SemModelSpec]:
    """Define only the retained SEM models used in the thesis.

    The final SEM set keeps four clean lipid/intermediate models plus the
    three HOMA-IR/glycemia models already documented in the current manuscript.
    Exploratory saturated, redundant or weak models are intentionally excluded.
    """
    anthropometry = keep_existing(df, ["IMC", "TT"])
    global_treatment = keep_existing(df, ["Escalade_therapeutique"])
    molecule_treatments = keep_existing(df, ["Metformine 500 mg", "Glibenclamide 5mg", "Insuline 10 UI"])
    lipid_mediators = keep_existing(df, ["Adipo HMW", "PCSK9", "HDL", "TG", "Non HDL"])
    ratio_sources = global_treatment + anthropometry
    molecule_anthro_sources = molecule_treatments + anthropometry
    nonhdl_sources = global_treatment + molecule_treatments + anthropometry

    models: list[SemModelSpec] = []
    source_lipid_mediators = keep_existing(df, ["PCSK9", "Adipo HMW", "HDL", "TG"])
    if anthropometry and source_lipid_mediators and "Ins (µUI/mL)" in df.columns:
        equations = [EquationSpec(mediator, anthropometry) for mediator in source_lipid_mediators]
        equations.append(EquationSpec("Ins (µUI/mL)", anthropometry + source_lipid_mediators))
        models.append(
            SemModelSpec(
                name="sem_clean_sources_insulin",
                description="Modele sans indices derives: adiposite -> lipides sources/biomarqueurs -> insuline observee",
                sources=anthropometry,
                covariates=[],
                mediators=source_lipid_mediators,
                outcome="Ins (µUI/mL)",
                equations=equations,
            )
        )

    if molecule_anthro_sources and "HDL" in df.columns:
        hdl_clean_mediators = keep_existing(df, ["PCSK9", "Adipo HMW", "TG"])
        if hdl_clean_mediators:
            equations = [EquationSpec(mediator, molecule_anthro_sources) for mediator in hdl_clean_mediators]
            equations.append(EquationSpec("HDL", molecule_anthro_sources + hdl_clean_mediators))
            models.append(
                SemModelSpec(
                    name="sem_clean_hdl_no_derived",
                    description="Modele HDL sans Non-HDL ni ratio TG-HDL dans les predicteurs",
                    sources=molecule_anthro_sources,
                    covariates=[],
                    mediators=hdl_clean_mediators,
                    outcome="HDL",
                    equations=equations,
                )
            )

    if nonhdl_sources and "Non HDL" in df.columns:
        nonhdl_clean_mediators = keep_existing(df, ["PCSK9", "Adipo HMW", "TG"])
        if nonhdl_clean_mediators:
            equations = [EquationSpec(mediator, nonhdl_sources) for mediator in nonhdl_clean_mediators]
            equations.append(EquationSpec("Non HDL", nonhdl_sources + nonhdl_clean_mediators))
            models.append(
                SemModelSpec(
                    name="sem_clean_nonhdl_no_components",
                    description="Modele Non-HDL sans HDL ni cholesterol total dans les predicteurs",
                    sources=nonhdl_sources,
                    covariates=[],
                    mediators=nonhdl_clean_mediators,
                    outcome="Non HDL",
                    equations=equations,
                )
            )

    if ratio_sources and "Ratio TG/HDL" in df.columns:
        ratio_clean_mediators = keep_existing(df, ["PCSK9", "Adipo HMW"])
        if ratio_clean_mediators:
            equations = [EquationSpec(mediator, ratio_sources) for mediator in ratio_clean_mediators]
            equations.append(EquationSpec("Ratio TG/HDL", ratio_sources + ratio_clean_mediators))
            models.append(
                SemModelSpec(
                    name="sem_clean_ratio_no_components",
                    description="Modele ratio TG-HDL sans TG ni HDL dans les predicteurs",
                    sources=ratio_sources,
                    covariates=[],
                    mediators=ratio_clean_mediators,
                    outcome="Ratio TG/HDL",
                    equations=equations,
                )
            )

    if global_treatment and lipid_mediators and "HOMA-IR" in df.columns:
        equations = [EquationSpec(mediator, global_treatment + anthropometry) for mediator in lipid_mediators]
        equations.append(EquationSpec("HOMA-IR", global_treatment + lipid_mediators + anthropometry))
        models.append(
            SemModelSpec(
                name="sem_global_homair",
                description="Intensite therapeutique observee -> biomarqueurs -> HOMA-IR",
                sources=global_treatment,
                covariates=anthropometry,
                mediators=lipid_mediators,
                outcome="HOMA-IR",
                equations=equations,
            )
        )

    if global_treatment and lipid_mediators and "Ins (µUI/mL)" in df.columns and "Glu" in df.columns:
        ins = ["Ins (µUI/mL)"]
        equations = [EquationSpec(mediator, global_treatment + anthropometry) for mediator in lipid_mediators]
        equations.append(EquationSpec(ins[0], global_treatment + lipid_mediators + anthropometry))
        equations.append(EquationSpec("Glu", global_treatment + lipid_mediators + ins + anthropometry))
        models.append(
            SemModelSpec(
                name="sem_global_glu",
                description="Intensite therapeutique observee -> biomarqueurs/insuline -> statut glycemique observe",
                sources=global_treatment,
                covariates=anthropometry,
                mediators=lipid_mediators + ins,
                outcome="Glu",
                equations=equations,
            )
        )

    if molecule_treatments and lipid_mediators and "HOMA-IR" in df.columns:
        equations = [EquationSpec(mediator, molecule_treatments + anthropometry) for mediator in lipid_mediators]
        equations.append(EquationSpec("HOMA-IR", molecule_treatments + lipid_mediators + anthropometry))
        models.append(
            SemModelSpec(
                name="sem_molecules_homair",
                description="Molecules antidiabetiques observees -> biomarqueurs -> HOMA-IR",
                sources=molecule_treatments,
                covariates=anthropometry,
                mediators=lipid_mediators,
                outcome="HOMA-IR",
                equations=equations,
            )
        )

    if not models:
        raise ValueError("No SEM model can be built with the available columns.")
    return models


def keep_existing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def model_columns(model: SemModelSpec) -> list[str]:
    columns: list[str] = []
    for equation in model.equations:
        columns.append(equation.outcome)
        columns.extend(equation.predictors)
    return list(dict.fromkeys(columns))


def prepare_sem_dataset(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sem_df = df[columns].copy()
    for col in sem_df.columns:
        sem_df[col] = pd.to_numeric(sem_df[col], errors="coerce")
    return sem_df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    # Les coefficients sont alors comparables: ils correspondent a des betas standardises.
    centered = df - df.mean()
    std = df.std(ddof=0).replace(0, np.nan)
    return centered.divide(std).fillna(0.0)


def fit_path_model(df: pd.DataFrame, model: SemModelSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_rows = []
    fit_rows = []
    for equation in model.equations:
        result = fit_ols(df[equation.outcome], df[equation.predictors])
        for predictor, beta, se, t_value, p_value in zip(
            equation.predictors,
            result["beta"],
            result["se"],
            result["t"],
            result["p"],
        ):
            parameter_rows.append(
                {
                    "equation": f"{equation.outcome} ~ {' + '.join(equation.predictors)}",
                    "outcome": equation.outcome,
                    "predictor": predictor,
                    "std_beta": beta,
                    "std_error": se,
                    "t_value": t_value,
                    "p_value": p_value,
                    "significant_0_05": bool(p_value < 0.05),
                }
            )
        fit_rows.append(
            {
                "equation": f"{equation.outcome} ~ {' + '.join(equation.predictors)}",
                "outcome": equation.outcome,
                "n": result["n"],
                "n_predictors": len(equation.predictors),
                "r_squared": result["r2"],
                "adjusted_r_squared": result["adj_r2"],
                "rmse": result["rmse"],
                "aic": result["aic"],
                "bic": result["bic"],
            }
        )
    return pd.DataFrame(parameter_rows), pd.DataFrame(fit_rows)


def fit_ols(y: pd.Series, x: pd.DataFrame) -> dict[str, object]:
    from scipy import stats

    y_values = y.to_numpy(dtype=float)
    x_values = x.to_numpy(dtype=float)
    n, p = x_values.shape
    design = np.column_stack([np.ones(n), x_values])
    coef = np.linalg.pinv(design.T @ design) @ design.T @ y_values
    pred = design @ coef
    resid = y_values - pred
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_values - y_values.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst else 0.0
    dof = max(n - p - 1, 1)
    sigma2 = sse / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t_values = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=dof))
    rmse = math.sqrt(sse / n)
    aic = n * math.log(max(sse / n, 1e-12)) + 2 * (p + 1)
    bic = n * math.log(max(sse / n, 1e-12)) + math.log(n) * (p + 1)
    return {
        "n": int(n),
        "beta": coef[1:],
        "se": se[1:],
        "t": t_values[1:],
        "p": p_values[1:],
        "r2": float(r2),
        "adj_r2": float(1 - (1 - r2) * (n - 1) / dof),
        "rmse": float(rmse),
        "aic": float(aic),
        "bic": float(bic),
    }


def compute_global_fit_indices(
    df: pd.DataFrame,
    model: SemModelSpec,
    parameters: pd.DataFrame,
    fit: pd.DataFrame,
) -> dict[str, object]:
    """Approximate covariance-based fit indices for the observed path model.

    The models are estimated equation by equation. These indices are therefore
    used as descriptive global diagnostics, alongside the local R2 values.
    """
    columns = model_columns(model)
    n = len(df)
    p = len(columns)
    if n < 5 or p < 2:
        return empty_global_fit(n, p)

    values = df[columns].to_numpy(dtype=float)
    sample_cov = np.cov(values, rowvar=False, bias=True)
    path_matrix = np.zeros((p, p), dtype=float)
    col_index = {col: idx for idx, col in enumerate(columns)}
    for row in parameters.itertuples(index=False):
        path_matrix[col_index[row.outcome], col_index[row.predictor]] = float(row.std_beta)

    endogenous = list(dict.fromkeys(eq.outcome for eq in model.equations))
    exogenous = [col for col in columns if col not in endogenous]
    residual_cov = np.zeros((p, p), dtype=float)
    for outcome in endogenous:
        rmse = float(fit.loc[fit["outcome"] == outcome, "rmse"].iloc[0])
        residual_cov[col_index[outcome], col_index[outcome]] = max(rmse**2, 1e-9)
    for left in exogenous:
        for right in exogenous:
            residual_cov[col_index[left], col_index[right]] = sample_cov[col_index[left], col_index[right]]

    identity = np.eye(p)
    try:
        implied_cov = np.linalg.inv(identity - path_matrix) @ residual_cov @ np.linalg.inv(identity - path_matrix).T
    except np.linalg.LinAlgError:
        return empty_global_fit(n, p)

    chi_square, discrepancy = covariance_discrepancy(sample_cov, implied_cov, n)
    baseline_cov = np.diag(np.diag(sample_cov))
    baseline_chi_square, _ = covariance_discrepancy(sample_cov, baseline_cov, n)

    n_paths = len(parameters)
    n_residual_variances = len(endogenous)
    n_exogenous_covariances = len(exogenous) * (len(exogenous) + 1) // 2
    free_parameters = n_paths + n_residual_variances + n_exogenous_covariances
    degrees_freedom = int(p * (p + 1) / 2 - free_parameters)
    baseline_df = int(p * (p - 1) / 2)

    cfi = np.nan
    if baseline_df > 0 and degrees_freedom > 0:
        denominator = max(baseline_chi_square - baseline_df, 1e-12)
        cfi = 1.0 - max(chi_square - degrees_freedom, 0.0) / denominator
        cfi = float(np.clip(cfi, 0.0, 1.0))

    rmsea = np.nan
    if degrees_freedom > 0:
        rmsea = math.sqrt(max((chi_square - degrees_freedom) / (degrees_freedom * max(n - 1, 1)), 0.0))

    offdiag = ~np.eye(p, dtype=bool)
    srmr = float(np.sqrt(np.mean((sample_cov[offdiag] - implied_cov[offdiag]) ** 2)))
    return {
        "n": int(n),
        "n_variables": int(p),
        "chi_square": float(chi_square),
        "degrees_freedom": int(degrees_freedom),
        "cfi": cfi,
        "rmsea": float(rmsea) if not np.isnan(rmsea) else np.nan,
        "srmr": srmr,
        "cfi_threshold": ">= 0.95",
        "rmsea_threshold": "<= 0.06",
        "srmr_threshold": "<= 0.08",
        "cfi_ok": bool(cfi >= 0.95) if not np.isnan(cfi) else False,
        "rmsea_ok": bool(rmsea <= 0.06) if not np.isnan(rmsea) else False,
        "srmr_ok": bool(srmr <= 0.08),
        "fit_note": (
            "Indices descriptifs pour path analysis observee estimee equation par equation."
            if degrees_freedom > 0
            else "Modele sature ou juste-identifie: CFI/RMSEA non interpretables, SRMR descriptif seulement."
        ),
        "discrepancy": float(discrepancy),
    }


def covariance_discrepancy(sample_cov: np.ndarray, implied_cov: np.ndarray, n: int) -> tuple[float, float]:
    sample_cov = nearest_positive_definite(sample_cov)
    implied_cov = nearest_positive_definite(implied_cov)
    p = sample_cov.shape[0]
    inv_implied = np.linalg.pinv(implied_cov)
    sign_s, logdet_s = np.linalg.slogdet(sample_cov)
    sign_i, logdet_i = np.linalg.slogdet(implied_cov)
    if sign_s <= 0 or sign_i <= 0:
        return np.nan, np.nan
    discrepancy = logdet_i + np.trace(sample_cov @ inv_implied) - logdet_s - p
    discrepancy = max(float(discrepancy), 0.0)
    return float(max(n - 1, 1) * discrepancy), discrepancy


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def empty_global_fit(n: int, p: int) -> dict[str, object]:
    return {
        "n": int(n),
        "n_variables": int(p),
        "chi_square": np.nan,
        "degrees_freedom": np.nan,
        "cfi": np.nan,
        "rmsea": np.nan,
        "srmr": np.nan,
        "cfi_threshold": ">= 0.95",
        "rmsea_threshold": "<= 0.06",
        "srmr_threshold": "<= 0.08",
        "cfi_ok": False,
        "rmsea_ok": False,
        "srmr_ok": False,
        "fit_note": "Indices non calculables pour cette specification.",
        "discrepancy": np.nan,
    }


def compute_effects(parameters: pd.DataFrame, model: SemModelSpec) -> pd.DataFrame:
    rows = []
    coef = {
        (row.predictor, row.outcome): float(row.std_beta)
        for row in parameters.itertuples(index=False)
    }
    for source in model.sources:
        direct = coef.get((source, model.outcome), 0.0)
        indirect_total = 0.0
        for mediator in model.mediators:
            if mediator == model.outcome:
                continue
            simple = coef.get((source, mediator), 0.0) * coef.get((mediator, model.outcome), 0.0)
            if simple:
                rows.append(
                    {
                        "source_variable": source,
                        "path": f"{source} -> {mediator} -> {model.outcome}",
                        "effect_type": "indirect_simple",
                        "effect": simple,
                    }
                )
                indirect_total += simple

            # Chemin medie en chaine utilise dans les modeles avec insuline.
            insulin = "Ins (µUI/mL)"
            if mediator != insulin and insulin in model.mediators:
                chain = (
                    coef.get((source, mediator), 0.0)
                    * coef.get((mediator, insulin), 0.0)
                    * coef.get((insulin, model.outcome), 0.0)
                )
                if chain:
                    rows.append(
                        {
                            "source_variable": source,
                            "path": f"{source} -> {mediator} -> {insulin} -> {model.outcome}",
                            "effect_type": "indirect_chain",
                            "effect": chain,
                        }
                    )
                    indirect_total += chain

        rows.extend(
            [
                {
                    "source_variable": source,
                    "path": f"{source} -> {model.outcome}",
                    "effect_type": "direct",
                    "effect": direct,
                },
                {
                    "source_variable": source,
                    "path": f"{source} -> mediateurs -> {model.outcome}",
                    "effect_type": "indirect_total",
                    "effect": indirect_total,
                },
                {
                    "source_variable": source,
                    "path": f"{source} -> total -> {model.outcome}",
                    "effect_type": "total",
                    "effect": direct + indirect_total,
                },
            ]
        )
    return pd.DataFrame(rows)


def build_summary_row(
    model: SemModelSpec,
    parameters: pd.DataFrame,
    fit: pd.DataFrame,
    global_fit: dict[str, object],
    effects: pd.DataFrame,
    n: int,
) -> dict[str, object]:
    outcome_fit = fit[fit["outcome"] == model.outcome].iloc[0]
    total_abs_indirect = effects.loc[effects["effect_type"].str.startswith("indirect"), "effect"].abs().sum()
    return {
        "model": model.name,
        "description": model.description,
        "n": n,
        "outcome": model.outcome,
        "outcome_r_squared": outcome_fit["r_squared"],
        "outcome_adjusted_r_squared": outcome_fit["adjusted_r_squared"],
        "outcome_rmse": outcome_fit["rmse"],
        "aggregate_aic": fit["aic"].sum(),
        "aggregate_bic": fit["bic"].sum(),
        "degrees_freedom": global_fit["degrees_freedom"],
        "cfi": global_fit["cfi"],
        "rmsea": global_fit["rmsea"],
        "srmr": global_fit["srmr"],
        "cfi_ok": global_fit["cfi_ok"],
        "rmsea_ok": global_fit["rmsea_ok"],
        "srmr_ok": global_fit["srmr_ok"],
        "n_significant_paths": int(parameters["significant_0_05"].sum()),
        "sum_abs_indirect_effects": total_abs_indirect,
    }


def write_model_spec(model: SemModelSpec, path: Path) -> None:
    lines = [model.description, ""]
    lines.extend(f"{eq.outcome} ~ {' + '.join(eq.predictors)}" for eq in model.equations)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_report(
    model: SemModelSpec,
    parameters: pd.DataFrame,
    fit: pd.DataFrame,
    global_fit: pd.DataFrame,
    effects: pd.DataFrame,
    path: Path,
) -> None:
    strongest = parameters.reindex(parameters["std_beta"].abs().sort_values(ascending=False).index).head(8)
    lines = [
        f"# {model.name}",
        "",
        model.description,
        "",
        "## Equations",
        "",
        *[f"- {eq.outcome} ~ {' + '.join(eq.predictors)}" for eq in model.equations],
        "",
        "## Fit local",
        "",
        markdown_table(fit.round(4)),
        "",
        "## Ajustement global descriptif",
        "",
        markdown_table(global_fit.round(4)),
        "",
        "## Chemins standardises les plus forts",
        "",
        markdown_table(strongest[["outcome", "predictor", "std_beta", "p_value", "significant_0_05"]].round(4)),
        "",
        "## Effets directs et indirects",
        "",
        markdown_table(effects.round(4)),
        "",
        "Note: ce modele est une path analysis observee. Les coefficients decrivent des associations structurees compatibles avec le graphe theorique. Ils ne demontrent pas une evolution longitudinale.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sem_summary_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Resume SEM",
        "",
        "Les modeles SEM sont estimes comme path analyses observees separees du pipeline ML.",
        "Ils testent des associations structurees entre anthropometrie, intensite therapeutique observee, biomarqueurs immuno-metaboliques, insulinoresistance et statut glycemique observe.",
        "Criteres indicatifs retenus: CFI >= 0,95 ; RMSEA <= 0,06 ; SRMR <= 0,08.",
        "",
        markdown_table(summary.round(4)),
        "",
        "Interpretation: un R2 eleve de l'equation finale indique que le bloc de chemins est associe a une part importante du statut observe. Les effets indirects non nuls indiquent une mediation exploratoire compatible avec les donnees transversales.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sem_excel_workbook(
    summary: pd.DataFrame,
    fit: pd.DataFrame,
    global_fit: pd.DataFrame,
    parameters: pd.DataFrame,
    effects: pd.DataFrame,
    path: Path,
) -> None:
    # Classeur de lecture rapide pour le memoire: tout le SEM dans un seul fichier.
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        fit.to_excel(writer, sheet_name="fit_metrics", index=False)
        global_fit.to_excel(writer, sheet_name="global_fit", index=False)
        parameters.to_excel(writer, sheet_name="parameters", index=False)
        parameters[parameters["significant_0_05"]].to_excel(writer, sheet_name="significant_paths", index=False)
        effects.to_excel(writer, sheet_name="effects", index=False)


def markdown_table(df: pd.DataFrame) -> str:
    # Petit rendu Markdown sans dependance optionnelle a tabulate.
    if df.empty:
        return "_Aucune ligne._"
    values = df.map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(values.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, sep, *rows])


def _require_matplotlib():
    import os
    import tempfile

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_sem_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    corr = df.corr(method="spearman")
    size = max(8, min(16, 0.55 * len(corr.columns)))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Matrice de correlation SEM")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_coefficients_plot(parameters: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    plot_df = parameters.copy()
    plot_df["label"] = plot_df["predictor"] + " -> " + plot_df["outcome"]
    plot_df = plot_df.reindex(plot_df["std_beta"].abs().sort_values(ascending=False).index).head(18)
    plot_df = plot_df.sort_values("std_beta")
    colors = np.where(plot_df["std_beta"] >= 0, "#4c78a8", "#e45756")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
    ax.barh(plot_df["label"], plot_df["std_beta"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Coefficient standardise")
    ax.set_title("Chemins SEM les plus forts")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_effects_plot(effects: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    plot_df = effects[effects["effect_type"].isin(["direct", "indirect_total", "total"])].copy()
    plot_df["label"] = plot_df["source_variable"] + " - " + plot_df["effect_type"]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(plot_df))))
    colors = {"direct": "#4c78a8", "indirect_total": "#f58518", "total": "#54a24b"}
    ax.barh(plot_df["label"], plot_df["effect"], color=[colors[t] for t in plot_df["effect_type"]])
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Effet standardise")
    ax.set_title("Effets directs, indirects et totaux")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_path_diagram(model: SemModelSpec, parameters: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    nodes = model.sources + model.covariates + model.mediators + [model.outcome]
    nodes = list(dict.fromkeys(nodes))
    positions = build_node_positions(model)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    for node in nodes:
        x, y = positions[node]
        box_color = "#f4f7fb"
        if node in model.sources:
            box_color = "#e9f2ff"
        elif node in model.mediators:
            box_color = "#eef8ed"
        elif node == model.outcome:
            box_color = "#fff2df"
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc=box_color, ec="#555555", lw=0.8),
        )

    for row in parameters.itertuples(index=False):
        start = positions[row.predictor]
        end = positions[row.outcome]
        color = "#2f6db3" if row.std_beta >= 0 else "#b33f3f"
        alpha = 0.95 if row.significant_0_05 else 0.35
        width = 0.8 + min(abs(row.std_beta) * 4, 3)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", lw=width, color=color, alpha=alpha, shrinkA=18, shrinkB=18),
        )
        label_x = (start[0] + end[0]) / 2
        label_y = (start[1] + end[1]) / 2
        ax.text(label_x, label_y, f"{row.std_beta:.2f}", fontsize=8, color=color, alpha=alpha)

    ax.set_title(model.description)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_smartpls_conceptual_diagram(path: Path) -> None:
    """Draw a SmartPLS-like conceptual model with constructs and indicators."""
    constructs = {
        "Traitement": {
            "pos": (0.10, 0.52),
            "indicators": ["Escalade", "Metformine", "Glibenclamide", "Insuline"],
        },
        "Anthropometrie": {
            "pos": (0.34, 0.78),
            "indicators": ["IMC", "TT"],
        },
        "Axe lipidique": {
            "pos": (0.52, 0.56),
            "indicators": ["HDL", "TG", "Non-HDL", "PCSK9"],
        },
        "Axe adipocytaire": {
            "pos": (0.52, 0.28),
            "indicators": ["Adipo HMW"],
        },
        "Axe insulinique": {
            "pos": (0.76, 0.54),
            "indicators": ["Ins", "HOMA-IR", "HOMAB%", "HOMAS%"],
        },
        "Reponse": {
            "pos": (0.92, 0.54),
            "indicators": ["Glu", "Controle glycemique"],
        },
    }
    arrows = [
        ("Traitement", "Axe lipidique"),
        ("Traitement", "Axe adipocytaire"),
        ("Traitement", "Axe insulinique"),
        ("Traitement", "Reponse"),
        ("Anthropometrie", "Axe lipidique"),
        ("Anthropometrie", "Axe adipocytaire"),
        ("Anthropometrie", "Axe insulinique"),
        ("Axe lipidique", "Axe insulinique"),
        ("Axe adipocytaire", "Axe insulinique"),
        ("Axe insulinique", "Reponse"),
        ("Axe lipidique", "Reponse"),
        ("Axe adipocytaire", "Reponse"),
    ]
    draw_smartpls_diagram(
        constructs=constructs,
        arrows=arrows,
        path=path,
        title="Modele conceptuel SEM type SmartPLS",
    )


def write_smartpls_model_diagram(model: SemModelSpec, parameters: pd.DataFrame, path: Path) -> None:
    """Draw the empirical SEM model in a SmartPLS-like grouped view."""
    if model.name == "sem_adiposity_homair":
        constructs = {
            "Anthropometrie": {"pos": (0.14, 0.56), "indicators": ["IMC", "TT"]},
            "Biomarqueurs": {"pos": (0.54, 0.56), "indicators": ["Adipo HMW", "PCSK9", "HDL", "TG", "Non-HDL"]},
            "Insulinoresistance": {"pos": (0.88, 0.56), "indicators": ["HOMA-IR"]},
        }
        arrows = [
            ("Anthropometrie", "Biomarqueurs"),
            ("Anthropometrie", "Insulinoresistance"),
            ("Biomarqueurs", "Insulinoresistance"),
        ]
    elif model.name == "sem_adiposity_glu":
        constructs = {
            "Anthropometrie": {"pos": (0.10, 0.56), "indicators": ["IMC", "TT"]},
            "Biomarqueurs": {"pos": (0.42, 0.56), "indicators": ["Adipo HMW", "PCSK9", "HDL", "TG", "Non-HDL"]},
            "Insuline": {"pos": (0.68, 0.56), "indicators": ["Ins"]},
            "Statut glycemique": {"pos": (0.91, 0.56), "indicators": ["Glu"]},
        }
        arrows = [
            ("Anthropometrie", "Biomarqueurs"),
            ("Anthropometrie", "Insuline"),
            ("Biomarqueurs", "Insuline"),
            ("Insuline", "Statut glycemique"),
            ("Biomarqueurs", "Statut glycemique"),
            ("Anthropometrie", "Statut glycemique"),
        ]
    elif model.name == "sem_biomarkers_ins_glu":
        constructs = {
            "Biomarqueurs": {"pos": (0.18, 0.56), "indicators": ["Adipo HMW", "PCSK9", "HDL", "TG", "Non-HDL"]},
            "Insuline": {"pos": (0.58, 0.56), "indicators": ["Ins"]},
            "Statut glycemique": {"pos": (0.88, 0.56), "indicators": ["Glu"]},
        }
        arrows = [
            ("Biomarqueurs", "Insuline"),
            ("Insuline", "Statut glycemique"),
            ("Biomarqueurs", "Statut glycemique"),
        ]
    elif model.name == "sem_lipid_ratio_homair":
        constructs = {
            "Sources observees": {"pos": (0.12, 0.56), "indicators": ["Escalade", "IMC", "TT"]},
            "Mediateurs cibles": {"pos": (0.54, 0.56), "indicators": ["Ratio TG/HDL", "PCSK9", "Adipo HMW"]},
            "Insulinoresistance": {"pos": (0.88, 0.56), "indicators": ["HOMA-IR"]},
        }
        arrows = [
            ("Sources observees", "Mediateurs cibles"),
            ("Sources observees", "Insulinoresistance"),
            ("Mediateurs cibles", "Insulinoresistance"),
        ]
    elif model.name == "sem_global_homair":
        constructs = {
            "Intensite observee": {"pos": (0.12, 0.52), "indicators": ["Escalade"]},
            "Anthropometrie": {"pos": (0.34, 0.78), "indicators": ["IMC", "TT"]},
            "Biomarqueurs": {"pos": (0.58, 0.52), "indicators": ["Adipo HMW", "PCSK9", "HDL", "TG", "Non-HDL"]},
            "Insulinoresistance": {"pos": (0.88, 0.52), "indicators": ["HOMA-IR"]},
        }
        arrows = [
            ("Intensite observee", "Biomarqueurs"),
            ("Anthropometrie", "Biomarqueurs"),
            ("Intensite observee", "Insulinoresistance"),
            ("Anthropometrie", "Insulinoresistance"),
            ("Biomarqueurs", "Insulinoresistance"),
        ]
    elif model.name == "sem_global_glu":
        constructs = {
            "Intensite observee": {"pos": (0.10, 0.52), "indicators": ["Escalade"]},
            "Anthropometrie": {"pos": (0.32, 0.78), "indicators": ["IMC", "TT"]},
            "Biomarqueurs": {"pos": (0.50, 0.52), "indicators": ["Adipo HMW", "PCSK9", "HDL", "TG", "Non-HDL"]},
            "Insuline": {"pos": (0.72, 0.52), "indicators": ["Ins"]},
            "Statut glycemique": {"pos": (0.91, 0.52), "indicators": ["Glu"]},
        }
        arrows = [
            ("Intensite observee", "Biomarqueurs"),
            ("Anthropometrie", "Biomarqueurs"),
            ("Biomarqueurs", "Insuline"),
            ("Intensite observee", "Insuline"),
            ("Anthropometrie", "Insuline"),
            ("Insuline", "Statut glycemique"),
            ("Biomarqueurs", "Statut glycemique"),
            ("Intensite observee", "Statut glycemique"),
            ("Anthropometrie", "Statut glycemique"),
        ]
    elif model.name == "sem_molecules_homair":
        constructs = {
            "Molecules observees": {"pos": (0.12, 0.52), "indicators": ["Metformine", "Glibenclamide", "Insuline"]},
            "Anthropometrie": {"pos": (0.34, 0.78), "indicators": ["IMC", "TT"]},
            "Biomarqueurs": {"pos": (0.58, 0.52), "indicators": ["Adipo HMW", "PCSK9", "HDL", "TG", "Non-HDL"]},
            "Insulinoresistance": {"pos": (0.88, 0.52), "indicators": ["HOMA-IR"]},
        }
        arrows = [
            ("Molecules observees", "Biomarqueurs"),
            ("Anthropometrie", "Biomarqueurs"),
            ("Molecules observees", "Insulinoresistance"),
            ("Anthropometrie", "Insulinoresistance"),
            ("Biomarqueurs", "Insulinoresistance"),
        ]
    else:
        constructs = {
            "Sources observees": {"pos": (0.14, 0.56), "indicators": shorten_indicators(model.sources)},
            "Mediateurs": {"pos": (0.54, 0.56), "indicators": shorten_indicators(model.mediators)},
            "Variable finale": {"pos": (0.88, 0.56), "indicators": shorten_indicators([model.outcome])},
        }
        arrows = [
            ("Sources observees", "Mediateurs"),
            ("Sources observees", "Variable finale"),
            ("Mediateurs", "Variable finale"),
        ]
    draw_smartpls_diagram(
        constructs=constructs,
        arrows=arrows,
        path=path,
        title=f"Representation type SmartPLS - {model.name}",
        arrow_labels=aggregate_construct_coefficients(model, parameters, constructs),
    )


def shorten_indicators(indicators: list[str], limit: int = 5) -> list[str]:
    aliases = {
        "Escalade_therapeutique": "Escalade",
        "Ins (µUI/mL)": "Ins",
        "Non HDL": "Non-HDL",
        "Metformine 500 mg": "Metformine",
        "Glibenclamide 5mg": "Glibenclamide",
        "Insuline 10 UI": "Insuline",
    }
    short = [aliases.get(indicator, indicator) for indicator in indicators]
    if len(short) <= limit:
        return short
    return short[: limit - 1] + ["..."]


def aggregate_construct_coefficients(
    model: SemModelSpec,
    parameters: pd.DataFrame,
    constructs: dict[str, dict[str, object]],
) -> dict[tuple[str, str], str]:
    """Aggregate path coefficients between construct groups for diagram labels."""
    indicator_to_construct = {}
    for construct, meta in constructs.items():
        for indicator in meta["indicators"]:
            indicator_to_construct[indicator] = construct
    aliases = {
        "Escalade": "Escalade_therapeutique",
        "Ins": "Ins (µUI/mL)",
        "Non-HDL": "Non HDL",
        "Metformine": "Metformine 500 mg",
        "Glibenclamide": "Glibenclamide 5mg",
        "Insuline": "Insuline 10 UI",
    }
    reverse_aliases = {value: key for key, value in aliases.items()}
    values: dict[tuple[str, str], list[float]] = {}
    for row in parameters.itertuples(index=False):
        predictor_key = reverse_aliases.get(row.predictor, row.predictor)
        outcome_key = reverse_aliases.get(row.outcome, row.outcome)
        source = indicator_to_construct.get(predictor_key)
        target = indicator_to_construct.get(outcome_key)
        if source and target and source != target:
            values.setdefault((source, target), []).append(float(row.std_beta))
    return {key: f"{np.mean(vals):.2f}" for key, vals in values.items()}


def draw_smartpls_diagram(
    constructs: dict[str, dict[str, object]],
    arrows: list[tuple[str, str]],
    path: Path,
    title: str,
    arrow_labels: dict[tuple[str, str], str] | None = None,
) -> None:
    plt = _require_matplotlib()
    arrow_labels = arrow_labels or {}
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=18)

    for source, target in arrows:
        sx, sy = constructs[source]["pos"]
        tx, ty = constructs[target]["pos"]
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color="#505050", lw=1.7, shrinkA=36, shrinkB=36),
        )
        label = arrow_labels.get((source, target))
        if label:
            ax.text((sx + tx) / 2, (sy + ty) / 2 + 0.025, label, ha="center", va="center", fontsize=9, color="#333333")

    for name, meta in constructs.items():
        x, y = meta["pos"]
        ellipse = plt.matplotlib.patches.Ellipse(
            (x, y),
            width=0.15,
            height=0.10,
            facecolor="#e8f1ff",
            edgecolor="#2f5d8c",
            lw=1.4,
        )
        ax.add_patch(ellipse)
        ax.text(x, y, name, ha="center", va="center", fontsize=9, weight="bold")
        draw_indicators(ax, x, y, list(meta["indicators"]))

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_indicators(ax, x: float, y: float, indicators: list[str]) -> None:
    from matplotlib.patches import FancyBboxPatch

    offsets = indicator_offsets(len(indicators))
    for indicator, (dx, dy) in zip(indicators, offsets):
        ix, iy = x + dx, y + dy
        rect = FancyBboxPatch(
            (ix - 0.047, iy - 0.018),
            0.094,
            0.036,
            boxstyle="round,pad=0.008,rounding_size=0.004",
            facecolor="#ffffff",
            edgecolor="#777777",
            lw=0.8,
        )
        ax.add_patch(rect)
        ax.text(ix, iy, indicator, ha="center", va="center", fontsize=7)
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(ix, iy),
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8, shrinkA=8, shrinkB=28),
        )


def indicator_offsets(n: int) -> list[tuple[float, float]]:
    if n == 1:
        return [(0, -0.105)]
    if n == 2:
        return [(-0.07, -0.105), (0.07, -0.105)]
    if n == 3:
        return [(-0.085, -0.105), (0, -0.125), (0.085, -0.105)]
    if n == 4:
        return [(-0.095, 0.105), (0.095, 0.105), (-0.095, -0.105), (0.095, -0.105)]
    return [(-0.11, 0.11), (0, 0.13), (0.11, 0.11), (-0.07, -0.12), (0.07, -0.12)]


def build_node_positions(model: SemModelSpec) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(model.sources):
        positions[node] = (0.08, 0.75 - idx * 0.18)
    for idx, node in enumerate(model.covariates):
        positions[node] = (0.08, 0.25 - idx * 0.18)
    for idx, node in enumerate(model.mediators):
        y = 0.85 - idx * (0.7 / max(len(model.mediators) - 1, 1))
        positions[node] = (0.52, y)
    positions[model.outcome] = (0.92, 0.5)
    return positions
