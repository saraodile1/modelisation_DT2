from __future__ import annotations

"""Point d'entree independant pour l'approche bayesienne.

Ce script part du dataset feature-engineered final et estime des probabilites
individualisees pour les deux cibles principales du memoire.
"""

import argparse
from pathlib import Path

from pipeline.bayesian_modeling import run_bayesian_pipeline
from pipeline.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SEM-guided Bayesian pipeline.")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to the main pipeline YAML config.")
    parser.add_argument(
        "--input",
        default="output_pipeline/features/engineered_dataset.xlsx",
        help="Feature-engineered dataset used as Bayesian input.",
    )
    parser.add_argument(
        "--output",
        default="output_pipeline/bayesian",
        help="Directory where Bayesian artefacts are written.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.is_absolute():
        input_path = config.root / input_path
    if not output_dir.is_absolute():
        output_dir = config.root / output_dir

    result = run_bayesian_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        cfg=config.raw,
        random_state=config.random_state,
    )
    print(f"Bayesian pipeline completed: {result['output_dir']}")


if __name__ == "__main__":
    main()
