from __future__ import annotations

"""Tables d'analyse exploratoire reproductibles.

Le but n'est pas de remplacer toute l'analyse visuelle des notebooks, mais de
produire automatiquement les tableaux essentiels: dimensions, types, manquants,
descriptifs, distribution du profil et correlations.
"""

import pandas as pd
from pathlib import Path
import os
import tempfile


def build_eda_tables(df: pd.DataFrame, cleaned: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    # cleaned est utilise pour l'EDA principale car il contient deja les sentinelles
    # converties en NaN; df sert seulement a documenter la dimension brute.
    profile_col = cfg["schema"]["profile_column"]
    numeric = cleaned.select_dtypes(include="number")

    # Tables generales: dimensions, types, valeurs manquantes, statistiques numeriques.
    tables: dict[str, pd.DataFrame] = {}
    tables["shape"] = pd.DataFrame(
        [{"rows": df.shape[0], "columns_raw": df.shape[1], "columns_cleaned": cleaned.shape[1]}]
    )
    tables["dtypes"] = cleaned.dtypes.astype(str).rename("dtype").to_frame()
    tables["missingness"] = (
        cleaned.isna().mean().mul(100).round(2).rename("missing_percent").to_frame()
        .assign(missing_count=cleaned.isna().sum())
        .sort_values(["missing_percent", "missing_count"], ascending=False)
    )
    tables["numeric_summary"] = numeric.describe().T.round(3)

    # Distribution du profil glycemique descriptif, si disponible.
    if profile_col in cleaned.columns:
        counts = cleaned[profile_col].value_counts(dropna=False).rename("count").to_frame()
        counts["percent"] = (counts["count"] / len(cleaned) * 100).round(2)
        tables["profile_distribution"] = counts

    # Correlations de Spearman sur les variables importantes du memoire, robuste
    # aux distributions asymetriques observees dans l'EDA.
    corr_columns = [
        c for c in [
            "Age", "IMC", "TT", "PAS", "PAD", "Prot", "HDL", "Totc", "LDLc",
            "TG", "Non HDL", "PCSK9", "Glu", "Ins (µUI/mL)", "HOMAB%",
            "HOMAS%", "HOMA-IR", "Adipo HMW", "Ratio Tot-c/HDL",
            "Ratio LDL-c/HDL", "Ratio TG/HDL", "Ratio LogTG/HDL", "TNFa",
        ]
        if c in cleaned.columns
    ]
    if len(corr_columns) >= 2:
        tables["spearman_correlations"] = cleaned[corr_columns].corr(method="spearman").round(3)

    return tables


def write_eda_tables(tables: dict[str, pd.DataFrame], output_dir) -> None:
    # Chaque table est exportee en CSV pour lecture facile dans Excel ou dans un rapport.
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", encoding="utf-8-sig", index_label="variable")


def write_eda_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Generate clear EDA figures from exported tables."""
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "matplotlib is required for figures. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    if "spearman_correlations" in tables:
        corr = tables["spearman_correlations"].astype(float)
        size = max(8, min(18, 0.55 * len(corr.columns)))
        fig, ax = plt.subplots(figsize=(size, size))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title("Matrice de correlation de Spearman")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / "spearman_correlation_matrix.png", dpi=200)
        plt.close(fig)

    if "missingness" in tables:
        miss = tables["missingness"].head(20).copy()
        if "missing_percent" in miss.columns:
            fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(miss))))
            ax.barh(miss.index.astype(str), miss["missing_percent"].astype(float), color="#5b8fd9")
            ax.invert_yaxis()
            ax.set_xlabel("Valeurs manquantes (%)")
            ax.set_title("Top 20 des variables avec valeurs manquantes")
            fig.tight_layout()
            fig.savefig(output_dir / "missingness_top20.png", dpi=200)
            plt.close(fig)

    if "profile_distribution" in tables:
        dist = tables["profile_distribution"].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(dist.index.astype(str), dist["count"].astype(float), color="#5aa469")
        ax.set_ylabel("Effectif")
        ax.set_title("Distribution du profil glycemique")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(output_dir / "profile_distribution.png", dpi=200)
        plt.close(fig)
