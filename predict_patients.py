from __future__ import annotations

"""Predire le controle glycemique et l'escalade therapeutique pour un dataset."""

import argparse
from pathlib import Path

from pipeline.config import load_config
from pipeline.io import read_table
from pipeline.prediction import build_dual_prediction_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict glycemic control and therapeutic escalation for new patients."
    )
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline YAML config.")
    parser.add_argument("--input", required=True, help="Excel or CSV file containing patient markers.")
    parser.add_argument(
        "--output",
        default="output_pipeline/predictions/patient_predictions.xlsx",
        help="Output Excel file with both predictions.",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "random_forest"],
        help="Trained model family to use for prediction.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config.raw
    patient_df = read_table(Path(args.input))
    result = build_dual_prediction_table(
        patient_df,
        cfg,
        models_dir=config.root / cfg["paths"]["output_dir"] / "models",
        model_name=args.model,
    )

    output_path = config.root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False)
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
