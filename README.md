# Pipeline officiel de la partie pratique du memoire

Ce dossier separe clairement deux phases:

- `Nettoyage/`: nettoyage initial historique, realise avant le pipeline automatise;
- pipeline officiel: traitement automatise a partir du dataset deja structure.

Le point de depart officiel est:

```text
data/structured/classification_profile_glicemique.xlsx
```

## Commandes

Installer ou mettre a jour les dependances:

```powershell
python -m pip install -r requirements.txt
```

Verifier que les scripts Python compilent:

```powershell
python -B -c "import ast, pathlib; files=[pathlib.Path('run_pipeline.py'), pathlib.Path('predict_patients.py'), *pathlib.Path('pipeline').glob('*.py')]; [ast.parse(p.read_text(encoding='utf-8')) for p in files]; print('Syntax OK')"
```

Verifier le schema, le nettoyage analytique et l'EDA sans lancer l'imputation:

```powershell
python run_pipeline.py --config config/pipeline.yaml --stop-after-pre-imputation
```

Relancer tout le pipeline avec une nouvelle imputation MissForest:

```powershell
python run_pipeline.py --config config/pipeline.yaml
```

Relancer rapidement en reutilisant la derniere imputation MissForest:

```powershell
python run_pipeline.py --config config/pipeline.yaml --skip-imputation
```

Predire le controle glycemique et l'escalade therapeutique avec XGBoost:

```powershell
python predict_patients.py --config config/pipeline.yaml --input output_pipeline/imputed/missforest_dataset.xlsx --output output_pipeline/predictions/patient_predictions.xlsx --model xgboost
```

Predire avec Random Forest:

```powershell
python predict_patients.py --config config/pipeline.yaml --input output_pipeline/imputed/missforest_dataset.xlsx --output output_pipeline/predictions/patient_predictions_random_forest.xlsx --model random_forest
```

Lancer le pipeline SEM separe, apres generation de `engineered_dataset.xlsx`:

```powershell
python run_sem_pipeline.py --config config/pipeline.yaml
```

Lancer le pipeline bayesien guide par les chemins SEM retenus:

```powershell
python run_bayesian_pipeline.py --config config/pipeline.yaml
```

## Ce que le pipeline automatise

1. validation du schema;
2. nettoyage analytique pre-imputation;
3. EDA reproductible;
4. exclusion des colonnes trop manquantes;
5. retrait puis recalcul des variables derivees;
6. imputation MissForest;
7. construction de deux cibles ML;
8. ajout de `LDLc_Friedewald` par la formule `Totc - HDL - TG/5`;
9. modele reduit sans ratios/conversions redondants, age et variables encodees sexe/HTA;
10. comparaison Random Forest / XGBoost en validation croisee;
11. calcul des metriques cliniques: AUC ROC, AUC Precision-Recall, accuracy,
    balanced accuracy, sensibilite, specificite, precision, F1-score, Brier score
    et matrice de confusion;
12. sauvegarde des modeles finaux;
13. prediction sur un tableau de patients;
14. generation automatique des figures EDA, courbes ROC, courbes
    Precision-Recall, courbes de calibration, matrices de confusion et
    graphiques d'importance des variables.

## Cibles ML

| Cible | Statut | Interpretation |
|---|---|---|
| `Controle_glycemique_binaire` | Principale | 0 = normoglycemie ou hypoglycemie; 1 = prediabete ou hyperglycemie/diabete |
| `Escalade_therapeutique` | Secondaire | 0 = aucun traitement ou monotherapie; 1 = bitherapie ou tritherapie |

Important: ces deux cibles sont transversales. Elles ne mesurent pas directement
une reponse therapeutique longitudinale avant/apres traitement, mais elles
approchent deux dimensions utiles: l'etat glycemique observe et l'intensite
therapeutique observee.

## Sorties principales

```text
output_pipeline/tables/
output_pipeline/imputed/missforest_dataset.xlsx
output_pipeline/imputed/missforest_dataset.pkl
output_pipeline/features/engineered_dataset.xlsx
output_pipeline/figures/eda/
output_pipeline/features/Controle_glycemique_binaire/
output_pipeline/features/Escalade_therapeutique/
output_pipeline/models/model_comparison_all_targets.csv
output_pipeline/models/model_comparison_all_targets.png
output_pipeline/models/Controle_glycemique_binaire/
output_pipeline/models/Escalade_therapeutique/
output_pipeline/predictions/patient_predictions.xlsx
output_pipeline/sem/
output_pipeline/bayesian/
output_pipeline/reports/run_summary.md
```

Dans chaque dossier de modele par cible, les fichiers les plus importants sont:

```text
model_comparison.csv
random_forest_cross_validation_metrics.csv
xgboost_cross_validation_metrics.csv
random_forest_holdout_clinical_metrics.csv
xgboost_holdout_clinical_metrics.csv
random_forest_holdout_roc_curve.png
xgboost_holdout_roc_curve.png
random_forest_holdout_precision_recall_curve.png
xgboost_holdout_precision_recall_curve.png
random_forest_holdout_confusion_matrix.png
xgboost_holdout_confusion_matrix.png
random_forest_holdout_calibration_curve.png
xgboost_holdout_calibration_curve.png
```

## Sorties SEM

Le pipeline SEM est volontairement separe du ML. Il teste des modeles de type
path analysis observee autour de la logique:

```text
sources observees -> biomarqueurs -> statut metabologique observe
```

Les principales sorties sont:

```text
output_pipeline/sem/sem_dataset.xlsx
output_pipeline/sem/sem_model_summary.csv
output_pipeline/sem/sem_results.xlsx
output_pipeline/sem/sem_summary_report.md
output_pipeline/sem/sem_parameters_all_models.csv
output_pipeline/sem/sem_effects_all_models.csv
output_pipeline/sem/sem_fit_metrics_all_models.csv
output_pipeline/sem/sem_global_fit_metrics_all_models.csv
output_pipeline/sem/figures/sem_correlation_heatmap.png
output_pipeline/sem/sem_clean_hdl_no_derived/
output_pipeline/sem/sem_clean_nonhdl_no_components/
output_pipeline/sem/sem_clean_ratio_no_components/
output_pipeline/sem/sem_clean_sources_insulin/
output_pipeline/sem/sem_global_homair/
output_pipeline/sem/sem_global_glu/
output_pipeline/sem/sem_molecules_homair/
```

Dans chaque modele SEM, lire:

```text
sem_model_spec.txt
sem_parameters.csv
sem_fit_metrics.csv
sem_effects.csv
sem_report.md
figures/sem_paths.png
figures/sem_standardized_coefficients.png
figures/sem_direct_indirect_total_effects.png
```

## Sorties bayesiennes

Le pipeline bayesien est une extension probabiliste guidee par les blocs SEM.
Il compare plusieurs modeles candidats par blocs biologiques, puis estime, pour
chaque patient, une probabilite individualisee accompagnee d'un intervalle
credible pour:

- `Controle_glycemique_binaire`;
- `Escalade_therapeutique`.

Les principales sorties sont:

```text
output_pipeline/bayesian/bayesian_results.xlsx
output_pipeline/bayesian/bayesian_model_summary.csv
output_pipeline/bayesian/bayesian_best_models.csv
output_pipeline/bayesian/bayesian_coefficients_all_targets.csv
output_pipeline/bayesian/bayesian_patient_predictions_all_targets.csv
output_pipeline/bayesian/bayesian_metrics_all_targets.csv
output_pipeline/bayesian/controle_sem_full/
output_pipeline/bayesian/controle_parsimonious_sem/
output_pipeline/bayesian/escalade_sem_full/
output_pipeline/bayesian/escalade_lipid_profile/
```

Dans chaque dossier bayesien par cible, lire:

```text
bayesian_coefficients.csv
bayesian_metrics.csv
bayesian_cross_validation_metrics.csv
bayesian_patient_predictions.xlsx
bayesian_holdout_predictions.csv
figures/bayesian_coefficients.png
figures/bayesian_calibration.png
figures/bayesian_probability_distribution.png
figures/bayesian_roc_curve.png
figures/bayesian_confusion_matrix.png
```

Les anciennes sorties MICE sont archivees dans:

```text
archive_ancien_travail/ancienne_imputation_mice/
```
