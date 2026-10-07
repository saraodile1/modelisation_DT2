from __future__ import annotations

"""Modelisation supervisee et comparaison de modeles.

Le module compare les classifieurs declares dans `config/pipeline.yaml` sur les
memes donnees imputees. La validation croisee stratifiee est le resultat
principal; le split train/test simple est conserve comme controle secondaire.
"""

from pathlib import Path
from typing import Callable
import os
import tempfile

import numpy as np
import pandas as pd
import pickle


ModelFactory = Callable[[int], object]


def run_random_forest(feature_data: dict[str, object], cfg: dict, output_dir: Path, random_state: int) -> dict:
    """Run configured classifiers for every target built by feature engineering."""
    if "targets" not in feature_data:
        return _run_configured_models(feature_data, cfg, output_dir, random_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_target_metrics = {}
    comparison_rows = []
    targets: dict[str, dict[str, object]] = feature_data["targets"]  # type: ignore[assignment]

    for target_name, target_data in targets.items():
        target_output_dir = output_dir / target_name
        metrics = _run_configured_models(target_data, cfg, target_output_dir, random_state)
        all_target_metrics[target_name] = metrics
        for model_name, model_metrics in metrics["models"].items():
            comparison_rows.append(
                {
                    "target": target_name,
                    "target_kind": target_data.get("target_kind", target_name),
                    "model": model_name,
                    "cv_auc_mean": model_metrics.get("cv_auc_mean"),
                    "cv_auc_std": model_metrics.get("cv_auc_std"),
                    "cv_auc_pr_mean": model_metrics.get("cv_auc_pr_mean"),
                    "cv_accuracy_mean": model_metrics.get("cv_accuracy_mean"),
                    "cv_accuracy_std": model_metrics.get("cv_accuracy_std"),
                    "cv_balanced_accuracy_mean": model_metrics.get("cv_balanced_accuracy_mean"),
                    "cv_sensitivity_mean": model_metrics.get("cv_sensitivity_mean"),
                    "cv_specificity_mean": model_metrics.get("cv_specificity_mean"),
                    "cv_precision_mean": model_metrics.get("cv_precision_mean"),
                    "cv_f1_mean": model_metrics.get("cv_f1_mean"),
                    "cv_brier_score_mean": model_metrics.get("cv_brier_score_mean"),
                    "holdout_auc": model_metrics.get("auc_pooled"),
                    "holdout_accuracy": model_metrics.get("accuracy_pooled"),
                    "holdout_auc_pr": model_metrics.get("auc_pr_pooled"),
                    "holdout_balanced_accuracy": model_metrics.get("balanced_accuracy_pooled"),
                    "holdout_sensitivity": model_metrics.get("sensitivity_pooled"),
                    "holdout_specificity": model_metrics.get("specificity_pooled"),
                    "holdout_precision": model_metrics.get("precision_pooled"),
                    "holdout_f1": model_metrics.get("f1_pooled"),
                    "holdout_brier_score": model_metrics.get("brier_score_pooled"),
                }
            )

    global_comparison = pd.DataFrame(comparison_rows).sort_values(
        ["target", "cv_auc_mean"], ascending=[True, False]
    )
    global_comparison.to_csv(
        output_dir / "model_comparison_all_targets.csv", index=False, encoding="utf-8-sig"
    )
    write_all_targets_comparison_plot(
        global_comparison,
        output_dir / "model_comparison_all_targets.png",
    )

    primary_target = cfg["target"]["glycemic_control"]["name"]
    primary_metrics = all_target_metrics.get(primary_target, next(iter(all_target_metrics.values())))
    return {
        "targets": all_target_metrics,
        "primary_target": primary_target,
        "comparison_metric": "cv_auc_mean",
        # Compatibilite: le resume historique pointe vers la cible principale.
        **primary_metrics,
    }


def _run_configured_models(feature_data: dict[str, object], cfg: dict, output_dir: Path, random_state: int) -> dict:
    """Run all configured classifiers for one target."""
    try:
        from sklearn.model_selection import StratifiedKFold, train_test_split
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "scikit-learn is required for modeling. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    x_list: list[pd.DataFrame] = feature_data["X_list"]  # type: ignore[assignment]
    y_list: list[pd.Series] = feature_data["y_list"]  # type: ignore[assignment]
    predictors: list[str] = feature_data["predicteurs"]  # type: ignore[assignment]

    output_dir.mkdir(parents=True, exist_ok=True)
    classifiers = cfg["modeling"].get("classifiers", [cfg["modeling"].get("classifier", "random_forest")])
    cv_cfg = cfg["modeling"].get("cross_validation", {})
    split_indices = build_train_test_indices(y_list[0], cfg, random_state)

    model_metrics: dict[str, dict] = {}
    comparison_rows = []

    for model_name in classifiers:
        model_output_prefix = model_name
        factory = make_model_factory(model_name, cfg, random_state)

        cv_metrics = {}
        if cv_cfg.get("enabled", True):
            cv_metrics = run_cross_validation(
                x_list=x_list,
                y_list=y_list,
                predictors=predictors,
                model_factory=factory,
                model_name=model_output_prefix,
                output_dir=output_dir,
                n_splits=int(cv_cfg.get("n_splits", 5)),
                random_state=random_state,
            )

        holdout_metrics = run_holdout_evaluation(
            x_list=x_list,
            y_list=y_list,
            predictors=predictors,
            model_factory=factory,
            model_name=model_output_prefix,
            output_dir=output_dir,
            split_indices=split_indices,
        )

        metrics = {**holdout_metrics, **cv_metrics}
        save_final_model(
            x_list=x_list,
            y_list=y_list,
            predictors=predictors,
            model_factory=factory,
            model_name=model_output_prefix,
            output_dir=output_dir,
            random_state=random_state,
        )
        model_metrics[model_name] = metrics
        comparison_rows.append(
            {
                "model": model_name,
                "cv_auc_mean": metrics.get("cv_auc_mean"),
                "cv_auc_std": metrics.get("cv_auc_std"),
                "cv_auc_pr_mean": metrics.get("cv_auc_pr_mean"),
                "cv_accuracy_mean": metrics.get("cv_accuracy_mean"),
                "cv_accuracy_std": metrics.get("cv_accuracy_std"),
                "cv_balanced_accuracy_mean": metrics.get("cv_balanced_accuracy_mean"),
                "cv_sensitivity_mean": metrics.get("cv_sensitivity_mean"),
                "cv_specificity_mean": metrics.get("cv_specificity_mean"),
                "cv_precision_mean": metrics.get("cv_precision_mean"),
                "cv_f1_mean": metrics.get("cv_f1_mean"),
                "cv_brier_score_mean": metrics.get("cv_brier_score_mean"),
                "holdout_auc": metrics.get("auc_pooled"),
                "holdout_accuracy": metrics.get("accuracy_pooled"),
                "holdout_auc_pr": metrics.get("auc_pr_pooled"),
                "holdout_balanced_accuracy": metrics.get("balanced_accuracy_pooled"),
                "holdout_sensitivity": metrics.get("sensitivity_pooled"),
                "holdout_specificity": metrics.get("specificity_pooled"),
                "holdout_precision": metrics.get("precision_pooled"),
                "holdout_f1": metrics.get("f1_pooled"),
                "holdout_brier_score": metrics.get("brier_score_pooled"),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_auc_mean", ascending=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    write_model_comparison_plot(comparison, output_dir / "model_comparison.png")

    best_model = comparison.iloc[0]["model"]
    return {
        "models": model_metrics,
        "best_model_cv_auc": best_model,
        "comparison_metric": "cv_auc_mean",
        # Aliases gardes pour le resume historique: ils pointent vers Random Forest
        # si present, sinon vers le meilleur modele.
        **model_metrics.get("random_forest", model_metrics[best_model]),
    }


def make_model_factory(model_name: str, cfg: dict, random_state: int) -> ModelFactory:
    """Create a model from its name and YAML parameters."""
    if model_name == "random_forest":
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("scikit-learn is required for Random Forest.") from exc

        rf_cfg = cfg["modeling"]["random_forest"]

        def factory(seed: int) -> object:
            return RandomForestClassifier(
                n_estimators=int(rf_cfg["n_estimators"]),
                max_depth=int(rf_cfg["max_depth"]),
                min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
                random_state=seed,
            )

        return factory

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "xgboost is required for the XGBoost model. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

        xgb_cfg = cfg["modeling"]["xgboost"]

        def factory(seed: int) -> object:
            return XGBClassifier(
                n_estimators=int(xgb_cfg["n_estimators"]),
                max_depth=int(xgb_cfg["max_depth"]),
                learning_rate=float(xgb_cfg["learning_rate"]),
                subsample=float(xgb_cfg["subsample"]),
                colsample_bytree=float(xgb_cfg["colsample_bytree"]),
                reg_lambda=float(xgb_cfg["reg_lambda"]),
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            )

        return factory

    raise ValueError(f"Unsupported classifier: {model_name!r}")


def build_train_test_indices(y: pd.Series, cfg: dict, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    return train_test_split(
        indices,
        test_size=float(cfg["modeling"]["test_size"]),
        stratify=y,
        random_state=random_state,
    )


def run_holdout_evaluation(
    x_list: list[pd.DataFrame],
    y_list: list[pd.Series],
    predictors: list[str],
    model_factory: ModelFactory,
    model_name: str,
    output_dir: Path,
    split_indices: tuple[np.ndarray, np.ndarray],
) -> dict:
    """Train/test evaluation kept as a secondary check."""
    idx_train, idx_test = split_indices
    probas_test = []
    aucs = []
    accuracies = []
    importances = []
    individual_rows = []

    for dataset_id, (x, y) in enumerate(zip(x_list, y_list), start=1):
        model = model_factory(10_000 + dataset_id)
        model.fit(x.iloc[idx_train], y.iloc[idx_train])
        proba = model.predict_proba(x.iloc[idx_test])[:, 1]
        binary_metrics = compute_binary_metrics(y.iloc[idx_test], proba)
        probas_test.append(proba)
        aucs.append(binary_metrics["auc"])
        accuracies.append(binary_metrics["accuracy"])
        individual_rows.append({"dataset_id": dataset_id, **binary_metrics})
        importances.append(get_feature_importance(model, predictors))

    y_test = y_list[0].iloc[idx_test]
    pooled_proba = np.mean(probas_test, axis=0)
    pooled_metrics = compute_binary_metrics(y_test, pooled_proba)
    write_roc_curve(
        y_true=y_test,
        proba=pooled_proba,
        auc_value=float(pooled_metrics["auc"]),
        title=f"Courbe ROC - {model_name}",
        path=output_dir / f"{model_name}_holdout_roc_curve.png",
    )
    write_precision_recall_curve(
        y_true=y_test,
        proba=pooled_proba,
        auc_pr_value=float(pooled_metrics["auc_pr"]),
        title=f"Courbe Precision-Recall - {model_name}",
        path=output_dir / f"{model_name}_holdout_precision_recall_curve.png",
    )
    write_confusion_matrix_plot(
        y_true=y_test,
        y_pred=(pooled_proba >= 0.5).astype(int),
        title=f"Matrice de confusion - {model_name}",
        path=output_dir / f"{model_name}_holdout_confusion_matrix.png",
    )
    write_calibration_curve(
        y_true=y_test,
        proba=pooled_proba,
        title=f"Calibration des probabilites - {model_name}",
        path=output_dir / f"{model_name}_holdout_calibration_curve.png",
    )

    pd.DataFrame(individual_rows).to_csv(
        output_dir / f"{model_name}_individual_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{f"{key}_pooled": value for key, value in pooled_metrics.items()}]).to_csv(
        output_dir / f"{model_name}_holdout_clinical_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_importance_summary(
        importances,
        predictors,
        output_dir / f"{model_name}_feature_importance.csv",
    )

    return {
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "auc_mean_individual": float(np.mean(aucs)),
        "auc_std_individual": float(np.std(aucs)),
        "auc_pooled": float(pooled_metrics["auc"]),
        "auc_pr_pooled": float(pooled_metrics["auc_pr"]),
        "accuracy_mean_individual": float(np.mean(accuracies)),
        "accuracy_std_individual": float(np.std(accuracies)),
        "accuracy_pooled": float(pooled_metrics["accuracy"]),
        "balanced_accuracy_pooled": float(pooled_metrics["balanced_accuracy"]),
        "sensitivity_pooled": float(pooled_metrics["sensitivity"]),
        "specificity_pooled": float(pooled_metrics["specificity"]),
        "precision_pooled": float(pooled_metrics["precision"]),
        "f1_pooled": float(pooled_metrics["f1"]),
        "brier_score_pooled": float(pooled_metrics["brier_score"]),
        "true_negative_pooled": int(pooled_metrics["true_negative"]),
        "false_positive_pooled": int(pooled_metrics["false_positive"]),
        "false_negative_pooled": int(pooled_metrics["false_negative"]),
        "true_positive_pooled": int(pooled_metrics["true_positive"]),
    }


def run_cross_validation(
    x_list: list[pd.DataFrame],
    y_list: list[pd.Series],
    predictors: list[str],
    model_factory: ModelFactory,
    model_name: str,
    output_dir: Path,
    n_splits: int,
    random_state: int,
) -> dict:
    """Run stratified K-fold cross-validation across imputed datasets."""
    from sklearn.model_selection import StratifiedKFold

    y_reference = y_list[0]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows = []
    fold_importances = []

    for fold_id, (idx_train, idx_test) in enumerate(splitter.split(x_list[0], y_reference), start=1):
        fold_probas = []
        fold_importances_per_dataset = []

        for dataset_id, (x, y) in enumerate(zip(x_list, y_list), start=1):
            model = model_factory(random_state + fold_id)
            model.fit(x.iloc[idx_train], y.iloc[idx_train])
            proba = model.predict_proba(x.iloc[idx_test])[:, 1]
            binary_metrics = compute_binary_metrics(y.iloc[idx_test], proba)

            fold_probas.append(proba)
            fold_importances_per_dataset.append(get_feature_importance(model, predictors))
            fold_rows.append(
                {
                    "fold": fold_id,
                    "dataset_id": dataset_id,
                    "n_train": int(len(idx_train)),
                    "n_test": int(len(idx_test)),
                    **binary_metrics,
                }
            )

        pooled_proba = np.mean(fold_probas, axis=0)
        y_test = y_reference.iloc[idx_test]
        pooled_fold_metrics = compute_binary_metrics(y_test, pooled_proba)
        fold_rows.append(
            {
                "fold": fold_id,
                "dataset_id": "pooled",
                "n_train": int(len(idx_train)),
                "n_test": int(len(idx_test)),
                **pooled_fold_metrics,
            }
        )
        fold_importances.append(np.mean(fold_importances_per_dataset, axis=0))

    cv_results = pd.DataFrame(fold_rows)
    cv_results.to_csv(
        output_dir / f"{model_name}_cross_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pooled_rows = cv_results[cv_results["dataset_id"] == "pooled"]
    write_importance_summary(
        fold_importances,
        predictors,
        output_dir / f"{model_name}_cross_validation_feature_importance.csv",
    )

    return {
        "cv_n_splits": int(n_splits),
        "cv_auc_mean": float(pooled_rows["auc"].mean()),
        "cv_auc_std": float(pooled_rows["auc"].std(ddof=0)),
        "cv_auc_pr_mean": float(pooled_rows["auc_pr"].mean()),
        "cv_auc_pr_std": float(pooled_rows["auc_pr"].std(ddof=0)),
        "cv_accuracy_mean": float(pooled_rows["accuracy"].mean()),
        "cv_accuracy_std": float(pooled_rows["accuracy"].std(ddof=0)),
        "cv_balanced_accuracy_mean": float(pooled_rows["balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(pooled_rows["balanced_accuracy"].std(ddof=0)),
        "cv_sensitivity_mean": float(pooled_rows["sensitivity"].mean()),
        "cv_sensitivity_std": float(pooled_rows["sensitivity"].std(ddof=0)),
        "cv_specificity_mean": float(pooled_rows["specificity"].mean()),
        "cv_specificity_std": float(pooled_rows["specificity"].std(ddof=0)),
        "cv_precision_mean": float(pooled_rows["precision"].mean()),
        "cv_precision_std": float(pooled_rows["precision"].std(ddof=0)),
        "cv_f1_mean": float(pooled_rows["f1"].mean()),
        "cv_f1_std": float(pooled_rows["f1"].std(ddof=0)),
        "cv_brier_score_mean": float(pooled_rows["brier_score"].mean()),
        "cv_brier_score_std": float(pooled_rows["brier_score"].std(ddof=0)),
    }


def compute_binary_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    """Compute model metrics useful for biomedical binary classification."""
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


def get_feature_importance(model: object, predictors: list[str]) -> np.ndarray:
    """Return feature importances when the model exposes them."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    return np.zeros(len(predictors), dtype=float)


def write_importance_summary(importances: list[np.ndarray], predictors: list[str], path: Path) -> None:
    importance_df = pd.DataFrame(importances, columns=predictors)
    importance_summary = pd.DataFrame(
        {
            "importance_mean": importance_df.mean(),
            "importance_std": importance_df.std(ddof=0),
        }
    ).sort_values("importance_mean", ascending=False)
    importance_summary.to_csv(path, encoding="utf-8-sig", index_label="predictor")
    write_feature_importance_plot(
        importance_summary,
        path.with_suffix(".png"),
        title=f"Importance des variables - {path.stem}",
    )


def save_final_model(
    x_list: list[pd.DataFrame],
    y_list: list[pd.Series],
    predictors: list[str],
    model_factory: ModelFactory,
    model_name: str,
    output_dir: Path,
    random_state: int,
) -> None:
    """Train one final model on all available rows and persist it for prediction."""
    if len(x_list) != 1:
        raise ValueError("Final model export currently expects one imputed dataset.")
    model = model_factory(random_state + 50_000)
    model.fit(x_list[0], y_list[0])
    payload = {
        "model": model,
        "predictors": predictors,
        "model_name": model_name,
    }
    with (output_dir / f"{model_name}_final_model.pkl").open("wb") as f:
        pickle.dump(payload, f)


def _require_matplotlib():
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "matplotlib is required for model figures. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return plt


def write_roc_curve(y_true: pd.Series, proba: np.ndarray, auc_value: float, title: str, path: Path) -> None:
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


def write_precision_recall_curve(
    y_true: pd.Series,
    proba: np.ndarray,
    auc_pr_value: float,
    title: str,
    path: Path,
) -> None:
    from sklearn.metrics import precision_recall_curve

    plt = _require_matplotlib()
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AUC PR = {auc_pr_value:.3f}", color="#8c5fbf", linewidth=2)
    ax.set_xlabel("Sensibilite / rappel")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_confusion_matrix_plot(y_true: pd.Series, y_pred: np.ndarray, title: str, path: Path) -> None:
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
    ax.set_ylabel("Classe reelle")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_calibration_curve(y_true: pd.Series, proba: np.ndarray, title: str, path: Path) -> None:
    from sklearn.calibration import calibration_curve

    plt = _require_matplotlib()
    fraction_positive, mean_predicted = calibration_curve(y_true, proba, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_predicted, fraction_positive, marker="o", color="#54a24b", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("Probabilite predite moyenne")
    ax.set_ylabel("Frequence observee de la classe positive")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_model_comparison_plot(comparison: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    plot_df = comparison.sort_values("model")
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, plot_df["cv_auc_mean"], width, label="AUC CV", color="#4c78a8")
    ax.bar(x + width / 2, plot_df["cv_accuracy_mean"], width, label="Accuracy CV", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"])
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison des modeles en validation croisee")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_all_targets_comparison_plot(comparison: pd.DataFrame, path: Path) -> None:
    plt = _require_matplotlib()
    labels = [f"{row.target}\n{row.model}" for row in comparison.itertuples()]
    x = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, comparison["cv_auc_mean"], color="#5b8fd9")
    ax.errorbar(
        x,
        comparison["cv_auc_mean"],
        yerr=comparison["cv_auc_std"],
        fmt="none",
        ecolor="#333333",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC moyenne en validation croisee")
    ax.set_title("Comparaison globale des cibles et modeles")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_feature_importance_plot(importance_summary: pd.DataFrame, path: Path, title: str) -> None:
    plt = _require_matplotlib()
    top = importance_summary.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index.astype(str), top["importance_mean"], color="#54a24b")
    ax.set_xlabel("Importance moyenne")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
