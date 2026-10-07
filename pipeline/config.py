from __future__ import annotations

"""Chargement de configuration et creation des dossiers de sortie.

Les YAML sont le contrat du pipeline: ils contiennent les chemins, colonnes,
seuils et conventions propres au dataset. Ce module evite de disperser ces
parametres dans les scripts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    # raw conserve toute la config YAML; root permet de resoudre les chemins
    # relatifs depuis la racine du projet.
    raw: dict[str, Any]
    root: Path

    @property
    def random_state(self) -> int:
        return int(self.raw.get("project", {}).get("random_state", 42))

    @property
    def input_dataset(self) -> Path:
        return self.root / self.raw["paths"]["input_dataset"]

    @property
    def output_dir(self) -> Path:
        return self.root / self.raw["paths"]["output_dir"]


def load_config(path: str | Path) -> PipelineConfig:
    # PyYAML est volontairement la seule dependance necessaire pour lire la config.
    config_path = Path(path).resolve()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "PyYAML is required to read config/pipeline.yaml. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid configuration file: {config_path}")
    return PipelineConfig(raw=raw, root=config_path.parent.parent)


def ensure_output_tree(config: PipelineConfig) -> dict[str, Path]:
    # Tous les artefacts generes sont ranges dans output_pipeline par type:
    # tables, imputations, features, modeles, rapports.
    base = config.output_dir
    paths = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "imputed": base / "imputed",
        "features": base / "features",
        "models": base / "models",
        "reports": base / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
