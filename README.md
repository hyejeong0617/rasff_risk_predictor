# RASFF Risk Predictor

*Retrospective Classification of EU Food Safety Notification Labels*

This scientific data science project investigates whether text and structured information in historical EU RASFF notifications can distinguish recorded **serious** decisions from other known labels. It connects temporal model evaluation, a frozen calibrated classifier and an interactive research dashboard, with explicit limits on food-safety interpretation.

**[Launch Interactive Streamlit App](https://rasffriskpredictor-rebuild.streamlit.app/)**

**[Explore the canonical Rebuild implementation](RASFF_Rebuild/)** — the only featured implementation. Explore historical notifications, predict recorded serious labels using the frozen model, and inspect evaluation results and limitations.

## Project overview

**Research question:** Can information recorded in a RASFF notification distinguish the `serious` label from the other known `risk_decision` labels?

The project uses a supplied historical snapshot of public Rapid Alert System for Food and Feed (RASFF) notification data spanning **2020-01-02 to 2025-10-31**. It includes food, feed and other supplied product types. The target is **1 = recorded serious; 0 = all other known recorded labels**.

This is retrospective label classification. A non-serious prediction **does not imply safe food**, and the app is not an operational risk assessment system.

## Key results

The final model is **Logistic Regression `raw_lr_C2` (C=2)** with TF-IDF, categorical encoding and separate sigmoid probability calibration. Performance below belongs to the **frozen 2024-11-26 to 2025-10-31 test period**, at a fixed decision threshold of 0.50.

| Evidence | Value |
|---|---:|
| Historical snapshot | **27,397 notifications** |
| Final test | **4,853 records** |
| Accuracy | **0.765** |
| Macro-F1 | **0.765** |
| ROC-AUC | **0.844** |
| Serious recall | **0.799** |

Sources: [data audit](RASFF_Rebuild/artifacts/data_audit.json), [frozen test report](RASFF_Rebuild/artifacts/test_report.json) and [saved test predictions](RASFF_Rebuild/artifacts/test_predictions.csv). Scores are rounded from saved artifacts, not recalculated for this README; they do not represent external or prospective validation.

## Data and methods

- **Provenance:** the inherited clean snapshot retains earlier manual edits and exclusions that cannot all be independently reconstructed. Malformed raw records are logged separately. See the [evidence audit](RASFF_Rebuild/audit/REPORT.md).
- **Six inputs:** `subject`, `origin`, `category`, `type`, `notifying_country` and `hazards`. `classification`, generated hazard fields and follow-up fields are excluded from the primary model.
- **Chronological evaluation:** whole observed calendar days define train / validation / calibration / test blocks of **14,668 / 4,748 / 3,128 / 4,853** records. See the [split manifest](RASFF_Rebuild/artifacts/split_manifest.json).
- **Preprocessing and selection:** TF-IDF vocabularies and categorical encoders are fitted within each training fold. Three expanding forward-CV folds compare eligible Logistic Regression, ComplementNB and XGBoost candidates by mean macro-F1, with average precision as tie-breaker. Validation is diagnostic; the selected pipeline then refits on train + validation. See [model selection](RASFF_Rebuild/02_Pipelines_and_Model_Selection.ipynb) and [saved comparisons](RASFF_Rebuild/artifacts/candidate_summary.csv).
- **Calibration and delivery:** the next block fits the sigmoid calibrator. [`final_bundle.joblib`](RASFF_Rebuild/artifacts/final_bundle.joblib) stores preprocessing, classifier, calibrator, input columns and threshold. Notebooks and the app use the same [`predict_bundle()`](RASFF_Rebuild/rasff_core.py) function; [saved verification](RASFF_Rebuild/artifacts/verification.json) records model/app parity checks.

## Interactive app

**[Launch Interactive Streamlit App](https://rasffriskpredictor-rebuild.streamlit.app/)**

1. **Explore notifications:** apply date, product, country and text filters; inspect descriptive charts and optionally download selected records.
2. **Predict recorded label:** enter the six model inputs and inspect the estimated probability of a recorded serious label, with interpretation caveats.
3. **Model & evaluation:** review frozen test metrics, the confusion matrix, temporal blocks and scientific limitations.

The app loads committed data and the saved bundle; it does not train models. The supplied URL returned HTTP 200 on 2026-11-04; this availability check is not a full hosted interaction test.

## Scientific interpretation and limitations

- **Labels are not harm probabilities.** Output estimates a recorded RASFF decision, not the probability of illness, contamination or food harm.
- **Class 0 is heterogeneous.** It includes not serious, undecided, potential risk, potentially serious and no risk. A negative prediction does not mean safe.
- **Field timing is unverified.** Public snapshot fields may have been updated after notification dates; related incidents may also span temporal blocks.
- **Cleaning history is incomplete.** Historical edits and exclusions, including the inherited correction to reference 2021.2621, are not fully reconstructed from original sources.
- **Validation is retrospective.** There is no external or prospective validation; historical calibration does not guarantee future reliability.
- **Notifications lack denominators.** RASFF records do not represent all marketed, imported or inspected products. Counts cannot establish population risk or country safety rankings.

The [seven-point audit](RASFF_Rebuild/audit/REPORT.md) documents evidence, caveats and reproducibility limits. Its findings describe the audit-time state; subsequent app presentation changes do not alter the frozen scientific results.

## Project contributions and tools

The implementation demonstrates scientific problem definition and feature selection; NLP/tabular preprocessing; temporal evaluation and model comparison; probability calibration and model serialization; Streamlit integration; and evidence-based reporting of scientific limitations.

Implementation and portfolio finalization include **AI-assisted coding and documentation**. These are project-level activities, not a claim that every component was independently authored by one person or that the model has peer-reviewed validation.

**Tools:** Python · pandas · scikit-learn · TF-IDF · Logistic Regression · Plotly · Streamlit · joblib. XGBoost and ComplementNB are comparison candidates, not the deployed classifier.

## Repository guide

| File | Purpose |
|---|---|
| [Rebuild README](RASFF_Rebuild/README.md) | Detailed setup and implementation guide |
| [01 — Data audit and EDA](RASFF_Rebuild/01_Data_Audit_and_EDA.ipynb) | Snapshot provenance, feature inventory and temporal blocks |
| [02 — Pipelines and selection](RASFF_Rebuild/02_Pipelines_and_Model_Selection.ipynb) | Fold-specific preprocessing and candidate comparison |
| [03 — Final evaluation and export](RASFF_Rebuild/03_Final_Evaluation_and_Model_Export.ipynb) | Calibration, frozen test evidence and model serialization |
| [04 — Interactive dashboard](RASFF_Rebuild/04_Interactive_Dashboard.ipynb) | Shared prediction contract and app workflow |
| [Shared model code](RASFF_Rebuild/rasff_core.py) | Canonical preprocessing and prediction functions |
| [Streamlit app](RASFF_Rebuild/app.py) | Current three-section portfolio application |
| [Evidence audit](RASFF_Rebuild/audit/REPORT.md) | Seven-point findings, scientific limitations and audit-time QA |
| [Saved evaluation outputs](RASFF_Rebuild/artifacts/) | Frozen metrics, predictions, split manifest and verification records |

**Archive:** `original_bootcamps_summary/` and `claude_rebuild_experimental/` are retained only as historical provenance; neither is the current portfolio implementation.

## Reproducibility and local execution

Use Python 3.12 in a fresh virtual environment. From the repository root:

```bash
python -m pip install -r RASFF_Rebuild/requirements.txt
python -m streamlit run RASFF_Rebuild/app.py
```

**No notebook execution or fresh training is required to run the app.** Keep the complete Rebuild folder, including prepared data, `rasff_core.py` and the saved bundle. Only load trusted joblib files.

Full training reproduction is a separate workflow described in the [technical guide](RASFF_Rebuild/README.md). Saved execution records cover notebooks 01–04, and some notebook outputs retain historical state; use the frozen artifacts for numerical claims. This documentation release did not rerun notebooks, recalculate metrics or perform a clean-room installation.

## Related portfolio projects

[AMR Genomics — Aeromonas](https://github.com/hyejeong0617/amr_genomics_aeromonas) · [Aeromonas Growth Predictor](https://github.com/hyejeong0617/aeromonas-growth-predictor) · [GitHub profile](https://github.com/hyejeong0617)
