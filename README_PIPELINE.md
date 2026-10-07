# Commandes du pipeline

Le pipeline officiel commence apres le nettoyage initial:

```text
data/structured/classification_profile_glicemique.xlsx
```

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Tests rapides

```powershell
python -B -c "import ast, pathlib; files=[pathlib.Path('run_pipeline.py'), pathlib.Path('predict_patients.py'), *pathlib.Path('pipeline').glob('*.py')]; [ast.parse(p.read_text(encoding='utf-8')) for p in files]; print('Syntax OK')"
```

```powershell
python run_pipeline.py --config config/pipeline.yaml --stop-after-pre-imputation
```

## Lancement complet

```powershell
python run_pipeline.py --config config/pipeline.yaml
```

## Lancement rapide sans refaire MissForest

```powershell
python run_pipeline.py --config config/pipeline.yaml --skip-imputation
```

## Prediction sur un tableau de patients

Avec XGBoost:

```powershell
python predict_patients.py --config config/pipeline.yaml --input output_pipeline/imputed/missforest_dataset.xlsx --output output_pipeline/predictions/patient_predictions.xlsx --model xgboost
```

Avec Random Forest:

```powershell
python predict_patients.py --config config/pipeline.yaml --input output_pipeline/imputed/missforest_dataset.xlsx --output output_pipeline/predictions/patient_predictions_random_forest.xlsx --model random_forest
```

## Pipeline SEM separe

```powershell
python run_sem_pipeline.py --config config/pipeline.yaml
```

Il part de:

```text
output_pipeline/features/engineered_dataset.xlsx
```

Et ecrit:

```text
output_pipeline/sem/
```

Le fichier d'entree de prediction doit contenir les marqueurs utilises par le
modele. S'il contient des valeurs manquantes, il faut d'abord passer par le
pipeline d'imputation MissForest.

## Sorties a lire

```text
output_pipeline/reports/run_summary.md
output_pipeline/models/model_comparison_all_targets.csv
output_pipeline/models/model_comparison_all_targets.png
output_pipeline/figures/eda/spearman_correlation_matrix.png
output_pipeline/features/engineered_dataset.xlsx
output_pipeline/imputed/missforest_dataset.xlsx
output_pipeline/predictions/patient_predictions.xlsx
output_pipeline/sem/sem_summary_report.md
output_pipeline/sem/figures/sem_correlation_heatmap.png
```

Pour interpreter les modeles, lire aussi dans chaque dossier de cible:

```text
model_comparison.csv
*_cross_validation_metrics.csv
*_holdout_clinical_metrics.csv
*_holdout_roc_curve.png
*_holdout_precision_recall_curve.png
*_holdout_confusion_matrix.png
*_holdout_calibration_curve.png
*_feature_importance.png
```
