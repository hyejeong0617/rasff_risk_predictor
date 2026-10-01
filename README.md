# RASFF Risk Predictor — from first ML project to audited rebuild

A retrospective machine-learning project using public EU RASFF notification data.

This repository intentionally preserves **three stages of the same project**: the original bootcamp work, an AI-assisted experimental rebuild, and the smaller audited rebuild that I now use as the portfolio reference.

> **Current portfolio version:** [`RASFF_Rebuild/`](./RASFF_Rebuild/)  
> It is a retrospective classifier of the recorded `serious` label. It is **not** a food-safety, recall, or regulatory-risk decision system.

## Why this repository has multiple versions

The first version was built during a data-science bootcamp and achieved about 87% test accuracy under its original evaluation design. I later revisited the project with AI coding assistance and added more advanced feature engineering, temporal evaluation ideas, and a richer Streamlit interface.

That second version looked more sophisticated, but the audit uncovered an important lesson: **more complex code is not automatically a more defensible model**. Some assumptions about prediction timing were stronger than the public data could support, preprocessing/model choices had become difficult to justify end-to-end, and the Streamlit predictor was not using the same trained model path described in the notebooks.

I therefore rebuilt the project again with a narrower question, a simpler architecture, explicit temporal blocks, preprocessing inside the fitted pipeline, separate calibration, and model/app parity checks.

## Repository map

| Folder | Role | Status |
|---|---|---|
| [`RASFF_Rebuild/`](./RASFF_Rebuild/) | Audited, executable rebuild and Streamlit app | **Current portfolio reference** |
| [`original_bootcamps_summary/`](./original_bootcamps_summary/) | Curated retrospective of the first bootcamp project | Historical evidence / learning record |
| [`claude_rebuild_experimental/`](./claude_rebuild_experimental/) | AI-assisted intermediate rebuild previously at repository root | Preserved experimental archive |

## Current rebuild

### Question

Can information recorded in a RASFF notification distinguish records labelled **`serious`** from the other known `risk_decision` labels?

The rebuild deliberately avoids claiming that all public fields were available before the official decision. Public snapshots may be updated after the notification date, so the project is framed as **retrospective label classification**, not proven pre-decision risk prediction.

### Default model inputs

The selected model uses six original fields:

- `subject`
- `origin`
- `category`
- `type`
- `notifying_country`
- `hazards`

Generated fields such as `Hazard_Type` and `simplified_hazard` are excluded from the primary model. `classification` is also excluded from the default prediction contract and tested separately as a late-information diagnostic.

### Evaluation design

- chronological calendar-day blocks: train / validation / calibration / final test
- forward cross-validation inside the training block
- TF-IDF and categorical encoders fit inside the sklearn pipeline
- candidate selection without using the final test period
- sigmoid calibration on a separate later block
- fixed decision threshold of 0.50
- final test written once and treated as frozen evidence

### Selected model

`raw_lr_C2` — Logistic Regression (`C=2`) with TF-IDF and one-hot encoded categorical inputs.

Final held-out test: **4,853 records**, 2024-11-26 to 2025-10-31.

| Metric | Score |
|---|---:|
| Accuracy | **0.765** |
| Macro F1 | **0.765** |
| Serious precision | **0.722** |
| Serious recall | **0.799** |
| Average precision | **0.803** |
| ROC-AUC | **0.844** |
| Brier score | **0.162** |
| Log loss | **0.496** |

These scores are **not directly comparable** with the original ~87% result because the feature representation, preprocessing, split design, and evaluation question changed.

### Verification

The rebuild includes explicit checks for:

- temporal block separation
- unseen token/category handling
- default feature scope
- save/reload prediction parity
- Streamlit filter behavior
- Streamlit prediction parity with the saved model bundle

See [`RASFF_Rebuild/artifacts/verification.json`](./RASFF_Rebuild/artifacts/verification.json).

## Classification diagnostic

A separate notebook adds only `classification` to the selected six-field architecture while keeping the rest of the comparison controlled.

This experiment is used to answer **how much signal the field adds**, not to promote it automatically into the primary model. Predictive value and operational availability are treated as separate questions.

See [`05_Classification_LateInfo_Ablation.ipynb`](./RASFF_Rebuild/05_Classification_LateInfo_Ablation.ipynb).

## Run the current version

```bash
cd RASFF_Rebuild
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run notebooks in order:

1. `01_Data_Audit_and_EDA.ipynb`
2. `02_Pipelines_and_Model_Selection.ipynb`
3. `03_Final_Evaluation_and_Model_Export.ipynb`
4. `04_Interactive_Dashboard.ipynb`
5. `05_Classification_LateInfo_Ablation.ipynb` — diagnostic extension

Launch the app:

```bash
python -m streamlit run app.py
```

For Korean execution notes, see [`RUN_GUIDE_KO.md`](./RASFF_Rebuild/RUN_GUIDE_KO.md).

## What I learned from rebuilding the project

The main result is not that Logistic Regression replaced XGBoost, or that a score moved from ~87% to ~76%.

The important change was methodological:

```text
first bootcamp project
        ↓
AI-assisted experimental rebuild
        ↓
audit assumptions and inference path
        ↓
redefine the prediction question
        ↓
chronology-aware, leakage-safe rebuild
        ↓
saved model contract
        ↓
same prediction path in the app
```

The project became more useful as a learning artifact when I could explain **what is predicted, when each input exists, how preprocessing is fit, where the test set is used, and whether the app really calls the evaluated model**.

## Data and scope

The supplied historical clean snapshot contains 27,397 records and inherits earlier manual edits/exclusions. The rebuild audits malformed raw rows but does not claim to reconstruct every historical cleaning decision from source.

Target definition:

- **1** = recorded `serious`
- **0** = every other known recorded label

Class 0 therefore does **not** mean safe.

RASFF Window is a notification database, not a denominator of all marketed or inspected food. Notification counts should not be interpreted as country, product, or population risk rates.

## Technology

Python · pandas · scikit-learn · XGBoost (comparison candidate) · Streamlit · Plotly · joblib

## Project status

**Portfolio 002: complete as an audited learning project.**

Future work should start from a real regulatory workflow and user need rather than continuing to optimize this frozen test set.
