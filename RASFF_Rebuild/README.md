# RASFF Rebuild — executable learning project

This package rebuilds the first project around correct data splitting, fold-specific preprocessing, model selection, probability calibration and a real saved-model app. Notebooks, comments and app labels are in English. See `RUN_GUIDE_KO.md` for Korean instructions.

## Quick start

Use Python 3.12 in a fresh virtual environment. Extract this complete folder first; do not copy a notebook alone.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name rasff-rebuild --display-name "RASFF Rebuild"
```

For app-only use, skip kernel registration and launch below from `RASFF_Rebuild/`. The committed prepared data and model bundle are sufficient; no notebook execution or retraining is required.

For a new reproduction cycle, notebooks 01–04 describe the sequential workflow with this folder as the working directory. Notebooks 01–02 can overwrite prepared/selection artifacts, and the existing final test must not be reused for tuning. Read the [evidence audit](audit/REPORT.md) before starting a new experiment.

```bash
python -m streamlit run app.py
```

## Files

- `01_Data_Audit_and_EDA.ipynb`: inherited snapshot provenance, malformed raw-row audit, missingness, feature inventory, exploratory plots and locked time blocks.
- `02_Pipelines_and_Model_Selection.ipynb`: fold-specific TF-IDF/one-hot pipelines, Logistic Regression, ComplementNB and XGBoost, forward cross-validation and optional matched ablations.
- `03_Final_Evaluation_and_Model_Export.ipynb`: fixed candidate refit, separate calibration, final test, reliability/error inspection and model serialization.
- `04_Interactive_Dashboard.ipynb`: input contract, launch instructions and optional UI smoke test.
- `rasff_core.py`: shared preprocessing, fitting and prediction; required alongside notebooks and app for serialization.
- `app.py`: Explorer, Predictor and Evaluation tabs.
- `verify_rebuild.py`: time boundaries, unknown-category handling, vocabulary leakage guard, serialization parity and real UI filter/prediction parity.
- `data/`: supplied clean and raw CSV, plus newly prepared dataset.
- `artifacts/`: actual executed outputs and trained bundle; no illustrative or manually invented scores.

## Design decisions

Target 1 means recorded `serious`; 0 includes all other known labels, including `undecided` and `potentially serious`. This is NOT a safe/unsafe classifier.

The default predictor uses original subject, original hazards, product category/type, origin and notifying country. All supplied product types are included. Classification, distribution and cooperation fields are not default predictors. Public snapshots may include information updated after the notification date: this remains retrospective classification, not proven pre-decision prediction.

TF-IDF is a local baseline with no paid API, external model download or LLM calls. Generated Hazard_Type and simplified_hazard remain in the source but are not used by default. Optional paired C=1 ablations report their added value without changing the already-selected primary model. Adoption would require a new development cycle. The current pipeline avoids target encoding rather than pretending to repair it through full-data preprocessing.

Calendar-day blocks: 60% training, 15% validation, 10% calibration, 15% final test. Row proportions differ. Expanding CV uses only the training block; macro F1 ranks candidates, average precision breaks ties. Validation is a diagnostic. The chosen pipeline refits on training plus validation. Sigmoid calibration uses only the next block. The threshold is fixed at 0.50. Test results are written once and loaded subsequently; never tune from them.

Explorer filters are descriptive and independent from Predictor. Prediction forms are generated from the saved feature contract, not a hard-coded three-input demo. Only development-period categories are offered as predictor options. Unknown countries/categories are handled by the saved pipeline; unseen information is not learned at prediction time.

## Verification and limitations

The supplied clean snapshot contains 27,397 records and inherits historical manual changes/exclusions. Raw malformed records are logged, not independently resolved. Auditing field counts finds both overlong and short records; its count need not match the original parser's skipped-row count.

The saved `artifacts/notebook_execution.json` records sequential in-process IPython execution of notebooks 01–04; it does not cover 05–06 or establish a fresh clean-room run. That recorded environment could not start a separate Jupyter kernel. Notebook 02 cell 10 retains a stale secondary output table; use `artifacts/candidate_summary.csv`, `artifacts/selected_spec.json` and `artifacts/test_report.json` for selection and final scores. Historical UI passes are recorded in `artifacts/verification.json`; they are not proof of a public deployment. See the [audit report](audit/REPORT.md) for independent checks and limitations. The full `verify_rebuild.py` script fits a model and writes artifacts, so it is not an inference-only smoke test.

The shipped held-out results are already visible: do not iterate toward better scores on this test period. A new modelling decision needs another untouched evaluation design. Calibration may improve or worsen transfer; use reliability plots and Brier/log loss rather than treating it as proof of trustworthy real-world probabilities. Scores are not directly comparable with the original 87.36% because representation and evaluation changed.

Only load trusted joblib/pickle files. The trained bundle was created with the pinned environment in requirements.txt; retraining is preferable when changing library versions.

Official references: [scikit-learn pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [Streamlit widgets](https://docs.streamlit.io/develop/api-reference/widgets/st.multiselect).
