from __future__ import annotations

"""Point d'entree independant pour la partie SEM.

Ce script ne relance pas le pipeline ML. Il part du dataset deja impute et
feature-engineered afin de tester des modeles de chemins entre sources
observees, biomarqueurs et statuts metaboliques.
"""

import argparse
from pathlib import Path

from pipeline.config import load_config
from pipeline.sem_modeling import run_sem_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent SEM/path-analysis pipeline.")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to the main pipeline YAML config.")
    parser.add_argument(
        "--input",
        default="output_pipeline/features/engineered_dataset.xlsx",
        help="Feature-engineered dataset used as SEM input.",
    )
    parser.add_argument(
        "--output",
        default="output_pipeline/sem",
        help="Directory where SEM artefacts are written.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.is_absolute():
        input_path = config.root / input_path
    if not output_dir.is_absolute():
        output_dir = config.root / output_dir

    result = run_sem_pipeline(input_path, output_dir, random_state=config.random_state)
    print(f"SEM pipeline completed: {result['output_dir']}")


if __name__ == "__main__":
    main()
