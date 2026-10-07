from __future__ import annotations

"""Imputation des valeurs manquantes.

La methode officielle actuelle est MissForest: les variables avec valeurs
manquantes sont imputees iterativement par Random Forest. Les variables derivees
sont recalculees apres imputation pour garder les relations mathematiques exactes
entre les variables sources.
"""

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .cleaning import CleaningResult, recalculate_derived_variables


def run_imputation(cleaning: CleaningResult, cfg: dict, output_dir: Path, random_state: int) -> list[pd.DataFrame]:
    """Dispatch vers la methode d'imputation declaree dans config/pipeline.yaml."""
    method = cfg["imputation"].get("method", "missforest")
    if method == "missforest":
        return run_missforest(cleaning, cfg, output_dir, random_state)
    raise ValueError(f"Unsupported imputation method: {method!r}")


def run_missforest(cleaning: CleaningResult, cfg: dict, output_dir: Path, random_state: int) -> list[pd.DataFrame]:
    """Run true MissForest on the pre-imputation table.

    MissForest produit un dataset complet. Pour garder la compatibilite avec la
    suite du pipeline, on retourne une liste contenant ce seul dataset.
    """
    data = cleaning.pre_imputation.copy()
    miss_cfg = cfg["imputation"]["missforest"]

    encoded, encoders, continuous_idx, categorical_idx = encode_for_missforest(data)
    imputed_array = impute_missforest(
        encoded.to_numpy(dtype=float),
        continuous_idx=continuous_idx,
        categorical_idx=categorical_idx,
        n_estimators=int(miss_cfg["n_estimators"]),
        max_depth=miss_cfg.get("max_depth"),
        max_iter=int(miss_cfg["max_iter"]),
        random_state=random_state,
    )

    imputed = decode_from_missforest(imputed_array, data.columns.tolist(), data, encoders)
    imputed = recalculate_derived_variables(imputed)

    redraw_cfg = cfg["imputation"].get("medication_positive_redraw", {})
    datasets = [imputed]
    if redraw_cfg.get("enabled", False):
        datasets = correct_positive_medication_doses(
            datasets=datasets,
            raw=cleaning.raw,
            columns=[c for c in redraw_cfg.get("columns", []) if c in cleaning.raw.columns],
            sentinel=cfg["cleaning"]["sentinel_missing"]["medication_taken_unknown_dose"],
            random_state=random_state,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "missforest_dataset.pkl").open("wb") as f:
        pickle.dump(datasets, f)
    datasets[0].to_excel(output_dir / "missforest_dataset.xlsx", index=False)
    imputation_report(cleaning.pre_imputation, datasets, cleaning.pre_imputation.columns[cleaning.pre_imputation.isna().any()].tolist()).to_csv(
        output_dir / "missforest_range_control.csv", index=False, encoding="utf-8-sig"
    )
    return datasets


def encode_for_missforest(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[object]], list[int], list[int]]:
    """Encode categories as numeric codes while preserving NaN.

    RandomForestClassifier needs numeric labels, so categorical columns are encoded
    before imputation and decoded after MissForest.
    """
    encoded = data.copy()
    encoders: dict[str, list[object]] = {}
    continuous_idx: list[int] = []
    categorical_idx: list[int] = []

    for idx, col in enumerate(encoded.columns):
        if pd.api.types.is_categorical_dtype(encoded[col]) or encoded[col].dtype == object:
            categories = list(pd.Series(encoded[col]).dropna().astype(str).unique())
            categories = sorted(categories)
            mapping = {value: code for code, value in enumerate(categories)}
            encoded[col] = encoded[col].astype(str).map(mapping)
            encoded.loc[data[col].isna(), col] = np.nan
            encoders[col] = categories
            categorical_idx.append(idx)
        else:
            encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
            continuous_idx.append(idx)

    return encoded, encoders, continuous_idx, categorical_idx


def decode_from_missforest(
    imputed_array: np.ndarray,
    columns: list[str],
    original: pd.DataFrame,
    encoders: dict[str, list[object]],
) -> pd.DataFrame:
    """Decode MissForest numeric matrix back to a DataFrame with original columns."""
    out = pd.DataFrame(imputed_array, columns=columns, index=original.index)
    for col, categories in encoders.items():
        codes = np.rint(out[col]).astype(int).clip(0, len(categories) - 1)
        out[col] = pd.Categorical([categories[code] for code in codes], categories=categories)
    for col in out.columns:
        if col not in encoders:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def impute_missforest(
    X_miss: np.ndarray,
    continuous_idx: List[int],
    categorical_idx: List[int],
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_iter: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """True MissForest imputation following Stekhoven & Buehlmann (2012)."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "scikit-learn is required for MissForest. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    overlap = set(continuous_idx) & set(categorical_idx)
    if overlap:
        raise ValueError(f"Indices {overlap} appear in both continuous and categorical lists.")

    X_imp = _initial_fill(X_miss, continuous_idx, categorical_idx)
    miss_counts = np.isnan(X_miss).sum(axis=0)
    active_vars = [int(j) for j in np.argsort(miss_counts) if miss_counts[j] > 0]
    if not active_vars:
        return X_imp

    gamma_cont_prev = np.inf
    gamma_cat_prev = np.inf
    X_best = X_imp.copy()

    for _ in range(max_iter):
        X_old = X_imp.copy()

        for j in active_vars:
            obs_rows = ~np.isnan(X_miss[:, j])
            mis_rows = np.isnan(X_miss[:, j])
            pred_cols = [k for k in range(X_imp.shape[1]) if k != j]
            X_train = X_imp[np.ix_(obs_rows, pred_cols)]
            y_train = X_imp[obs_rows, j]
            X_pred = X_imp[np.ix_(mis_rows, pred_cols)]

            if X_pred.shape[0] == 0:
                continue

            if j in categorical_idx:
                classes = np.unique(y_train.astype(int))
                if len(classes) < 2:
                    preds = np.full(X_pred.shape[0], classes[0], dtype=float)
                else:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    model.fit(X_train, y_train.astype(int))
                    preds = model.predict(X_pred).astype(float)
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_pred)

            X_imp[mis_rows, j] = preds

        gamma_cont = _gamma_continuous(X_imp, X_old, continuous_idx, miss_counts)
        gamma_cat = _gamma_categorical(X_imp, X_old, categorical_idx, miss_counts)
        cont_ok = (not continuous_idx) or (gamma_cont <= gamma_cont_prev)
        cat_ok = (not categorical_idx) or (gamma_cat <= gamma_cat_prev)

        if cont_ok and cat_ok:
            X_best = X_imp.copy()
            gamma_cont_prev = gamma_cont
            gamma_cat_prev = gamma_cat
        else:
            break

    return X_best


def _initial_fill(X_miss: np.ndarray, continuous_idx: List[int], categorical_idx: List[int]) -> np.ndarray:
    """Initial mean/mode fill required before the first MissForest iteration."""
    X = X_miss.astype(float, copy=True)
    for j in continuous_idx:
        col = X[:, j]
        fill = np.nanmean(col)
        if np.isnan(fill):
            fill = 0.0
        col[np.isnan(col)] = fill
        X[:, j] = col
    for j in categorical_idx:
        col = X[:, j]
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            fill = 0.0
        else:
            vals, counts = np.unique(observed, return_counts=True)
            fill = float(vals[np.argmax(counts)])
        col[np.isnan(col)] = fill
        X[:, j] = col
    return X


def _gamma_continuous(X_new: np.ndarray, X_old: np.ndarray, cont_idx: List[int], miss_counts: np.ndarray) -> float:
    """Continuous convergence criterion from Stekhoven & Buehlmann."""
    if not cont_idx:
        return 0.0
    num, den = 0.0, 0.0
    for j in cont_idx:
        if int(miss_counts[j]) == 0:
            continue
        diff = X_new[:, j] - X_old[:, j]
        num += float(np.sum(diff**2))
        den += float(np.sum(X_new[:, j]**2))
    return num / den if den > 1e-12 else 0.0


def _gamma_categorical(X_new: np.ndarray, X_old: np.ndarray, cat_idx: List[int], miss_counts: np.ndarray) -> float:
    """Categorical convergence criterion from Stekhoven & Buehlmann."""
    if not cat_idx:
        return 0.0
    total = sum(int(miss_counts[j]) for j in cat_idx)
    if total == 0:
        return 0.0
    disagree = sum(
        int(np.sum(X_new[:, j].astype(int) != X_old[:, j].astype(int)))
        for j in cat_idx
    )
    return disagree / total


def correct_positive_medication_doses(
    datasets: list[pd.DataFrame],
    raw: pd.DataFrame,
    columns: list[str],
    sentinel: int | float,
    random_state: int,
) -> list[pd.DataFrame]:
    # Cette fonction n'est active que si une convention de sentinelle existe.
    # Sans sentinelle, le pipeline ne suppose pas qu'il connait le statut de prise.
    corrected = [d.copy() for d in datasets]
    if sentinel is None:
        return corrected
    for col in columns:
        # Patients dont on sait qu'ils prennent le medicament, mais dont la dose
        # etait inconnue dans le dataset brut.
        patients_taken_unknown = raw.index[raw[col] == sentinel]
        positive_observed = raw.loc[raw[col] > 0, col].dropna().to_numpy()
        if len(patients_taken_unknown) == 0 or len(positive_observed) == 0:
            continue
        for i, d in enumerate(corrected):
            # Graine differente par jeu impute pour conserver une variabilite.
            rng = np.random.RandomState(random_state + 1000 + i)
            zero_mask = d.loc[patients_taken_unknown, col] == 0
            n_zero = int(zero_mask.sum())
            if n_zero:
                replacement = rng.choice(positive_observed, size=n_zero, replace=True)
                target_idx = patients_taken_unknown[zero_mask.to_numpy()]
                d.loc[target_idx, col] = replacement
    return corrected


def imputation_report(
    pre_imputation: pd.DataFrame,
    datasets: list[pd.DataFrame],
    variables_to_impute: list[str],
) -> pd.DataFrame:
    # Controle de plausibilite: MissForest peut predire une valeur continue hors
    # de la plage observee. Ce rapport signale ces cas pour inspection.
    rows = []
    for var in variables_to_impute:
        if not pd.api.types.is_numeric_dtype(pre_imputation[var]):
            continue
        observed = pre_imputation[var].dropna()
        combined = pd.concat([d[var] for d in datasets], ignore_index=True)
        rows.append(
            {
                "variable": var,
                "observed_min": observed.min(),
                "observed_max": observed.max(),
                "imputed_min": combined.min(),
                "imputed_max": combined.max(),
                "outside_observed_range": combined.min() < observed.min()
                or combined.max() > observed.max(),
            }
        )
    return pd.DataFrame(rows)
