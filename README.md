# RASFF Risk Predictor | Retrospective Notification Classification

**Food-safety data science · Temporal evaluation · NLP · Calibrated machine learning · Streamlit**

Can recorded RASFF notification information distinguish the **`serious` label** from other known risk-decision labels?

This project answers that retrospective question using a historical notification snapshot, a shared scikit-learn pipeline and an interactive saved-model app. **[`RASFF_Rebuild/`](RASFF_Rebuild/) is the sole current portfolio implementation.**

## Interactive Streamlit App

**[App source and local launch](RASFF_Rebuild/app.py)** · **[Setup guide](RASFF_Rebuild/README.md)** · **[Evidence audit](RASFF_Rebuild/audit/REPORT.md)**

The app provides three views:

- **Explore notifications:** filter historical records, inspect descriptive charts and download selected rows.
- **Predict recorded label:** enter the six fields used by the saved model.
- **Model & evaluation:** inspect the selected candidate, frozen test metrics and confusion matrix.

A public deployment URL has not been verified. Launch the app locally with the committed data and bundle; notebook execution and retraining are unnecessary.

> **Interpretation:** this is retrospective classification of recorded labels, not a validated food-safety, recall or regulatory decision system. “Other recorded labels” does not mean safe.

## Study scope and saved results

| Item | Verified evidence |
|---|---|
| Historical clean snapshot | **27,397 records**, 2020-01-02 to 2025-10-31; includes food, feed and other supplied types |
| Target | `1 = serious`; `0 = not serious, undecided, potential risk, potentially serious, no risk` |
| Selected model | **Logistic Regression `raw_lr_C2`**, C=2, followed by separate sigmoid calibration |
| Final test | **4,853 records**, 2024-11-26 to 2025-10-31 |
| Decision threshold | **0.50** |

| Calibrated final-test metric | Saved score |
|---|---:|
| Accuracy | **0.765** |
| Macro F1 | **0.765** |
| ROC-AUC | **0.844** |
| Serious precision | 0.722 |
| Serious recall | 0.799 |
| Average precision | 0.803 |
| Brier score | 0.162 |
| Log loss | 0.496 |

Source: **[frozen test report](RASFF_Rebuild/artifacts/test_report.json)** and [saved predictions](RASFF_Rebuild/artifacts/test_predictions.csv). Values are rounded from saved evaluation artifacts; the portfolio audit did not retrain the model or recalculate metrics. These are historical held-out results, not external or prospective validation.

## Methodology and project contribution

The implementation connects explicit temporal evaluation, preprocessing within fitted pipelines, a controlled feature diagnostic and the same prediction function in notebooks and Streamlit.

1. **Define the target and audit the snapshot.** Preserve the distinction between recorded seriousness and actual safety; record malformed raw rows and inherited cleaning limitations.
2. **Split by whole observed calendar days.** Train, validation, calibration and test contain 14,668 / 4,748 / 3,128 / 4,853 records. See the [split manifest](RASFF_Rebuild/artifacts/split_manifest.json).
3. **Compare candidates within development data.** Three expanding forward-CV folds rank eligible Logistic Regression, ComplementNB and XGBoost candidates by mean macro F1, with average precision as tie-breaker. Validation is a diagnostic.
4. **Keep learned preprocessing inside each pipeline.** Subject/hazard TF-IDF, origin tokens and categorical one-hot encoding are fitted with the corresponding training data. The final selected pipeline refits on train + validation; a separate later block fits the sigmoid calibrator.
5. **Export one prediction contract.** The [saved bundle](RASFF_Rebuild/artifacts/final_bundle.joblib) contains preprocessing, classifier, calibrator, input columns and threshold. The app calls the shared [`predict_bundle`](RASFF_Rebuild/rasff_core.py) function.

The six predictor fields are **`subject`, `origin`, `category`, `type`, `notifying_country`, `hazards`**. Generated hazard fields and follow-up fields are excluded from the primary contract.

### Late-information diagnostic

[Notebook 05](RASFF_Rebuild/05_Classification_LateInfo_Ablation.ipynb) clones the selected architecture and adds only `classification`. Its saved **uncalibrated validation** macro F1 rises from 0.769836 to 0.842444. This demonstrates additional retrospective signal; it does not establish that the field was available before the decision. `classification` remains excluded from the app, and the diagnostic does not replace the frozen final model or metrics.

## Run locally

The Rebuild guide specifies Python 3.12 and pinned dependencies. From a fresh clone:

```bash
git clone https://github.com/hyejeong0617/rasff_risk_predictor.git
cd rasff_risk_predictor/RASFF_Rebuild
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app uses shipped `data/prepared.csv` and saved artifacts. Keep `rasff_core.py` alongside the app because the serialized pipeline references it. Load only trusted joblib files.

[Rebuild setup and design](RASFF_Rebuild/README.md) · [Korean run guide](RASFF_Rebuild/RUN_GUIDE_KO.md) · [Recorded dependency versions](RASFF_Rebuild/tested_environment.json)

## Scientific limitations and reproducibility

- **Inherited provenance:** the clean snapshot retains earlier manual edits/exclusions, including reference 2021.2621. Its 27,397 rows and source hash are verified locally; the original cleaning history is not independently reconstructed.
- **Timing and dependence:** chronological separation and fold-specific preprocessing are implemented, but public snapshot fields may contain later updates. Related notifications may span blocks. This does not establish pre-decision validity or exclude every leakage route.
- **Notification bias:** the data lack a denominator of all marketed, imported or inspected products. Counts and model scores do not estimate population risk or country safety.
- **Label and calibration limits:** class 0 includes uncertain and potentially serious labels. Historical calibration does not guarantee reliable future probabilities. No external validation or uncertainty intervals are supplied.
- **Frozen evidence:** saved execution records cover notebooks 01–04; Notebook 02 retains a stale secondary output table. The authoritative selection and final metrics are the linked artifacts. The audit did not rerun notebooks or perform a clean-room dependency installation.

The **[seven-point audit](RASFF_Rebuild/audit/REPORT.md)** distinguishes implementation checks, saved verification results and new lightweight QA. Status: **READY WITH CAVEATS** as a retrospective learning portfolio.

## Repository guide

| Path | Purpose |
|---|---|
| [`RASFF_Rebuild/`](RASFF_Rebuild/) | Current notebooks, shared pipeline, app, data and frozen artifacts |
| [`RASFF_Rebuild/audit/REPORT.md`](RASFF_Rebuild/audit/REPORT.md) | Evidence, issue register, scientific limitations and QA outcomes |
| [`RASFF_Rebuild/06_Regulatory_Agent_Raw_Hazard_EDA.ipynb`](RASFF_Rebuild/06_Regulatory_Agent_Raw_Hazard_EDA.ipynb) | Separate exploratory case-selection extension; not a validated regulatory agent |

Archived learning history is preserved in [`original_bootcamps_summary/`](original_bootcamps_summary/) and [`claude_rebuild_experimental/`](claude_rebuild_experimental/). These folders are outside the current implementation and audit scope.

## Related projects

- [AMR Genomics — Aeromonas](https://github.com/hyejeong0617/amr_genomics_aeromonas)
- [Aeromonas Growth Predictor](https://github.com/hyejeong0617/aeromonas-growth-predictor)
- [Foodborne Outbreaks EDA](https://github.com/hyejeong0617/foodborne_outbreaks_eda)

[GitHub profile](https://github.com/hyejeong0617)
