# RASFF Rebuild — portfolio evidence audit

Audit date: 2026-11-04 (Europe/Berlin). Baseline: `62db156ed98a7780299d8bc28aee1b0baf7682fe`, branch `main`.

## Scope and decision

**READY WITH CAVEATS for a retrospective learning portfolio; not validated for prospective food-safety decisions.** No material P0 invalidating the published retrospective model or saved metrics was identified. P1 scientific limitations remain below.

The initial workspace was not a Git repository and contained no RASFF checkout. A fresh clone was inspected: `main...origin/main`, clean. No existing changes were overwritten. Audit scope is `RASFF_Rebuild/` and the root README. Historical implementation folders were not inspected. Notebook 06 is a separate exploratory case-selection extension, not evidence for primary-model performance.

This report was written before README changes. Evidence means inspected source, saved outputs or explicitly recorded checks, not a claim that training was reproduced. No training, notebook execution, dataset editing or metric recalculation was performed. Notebook cell numbers below are **one-based positions including markdown**, not execution counts. Paths are relative to `RASFF_Rebuild/` unless stated otherwise.

## Seven-point checklist

| # | Criterion | Status | Evidence and boundary | Confidence |
|---|---|---|---|---|
| 1 | Research question and target definition | VERIFIED | Notebook 01 cells 1, 7–8; `rasff_core.py:25–27,48–64,118–123`; `app.py:12–13,81–86`. Target is recorded `serious` vs five other known labels, not safe/unsafe. Prepared target labels independently checked. | High |
| 2 | Data provenance and cleaning | PARTIALLY VERIFIED | `artifacts/data_audit.json`; Notebook 01 cells 3–6; `data/CleanedDataset.csv`, `data/prepared.csv`; `artifacts/raw_malformed_rows.csv`. Row count, unique references and source hash verified. Original acquisition/export metadata, inherited exclusions and manual correction 2021.2621 are not independently reconstructed. | High for local integrity; limited for upstream provenance |
| 3 | Methods, model selection and contributions | VERIFIED | `rasff_core.py:103–195`; Notebook 02 cells 4, 6, 8; `artifacts/{cv_results,candidate_summary,validation_results}.csv`, `selected_spec.json`. Implemented contribution is a shared temporal pipeline, controlled candidate/feature comparisons and saved-model UI. Personal authorship is not independently established. | High for implementation |
| 4 | Results and metric traceability | VERIFIED | `artifacts/test_report.json`; Notebook 03 cell 4; `artifacts/test_predictions.csv`; source hash matches local clean CSV. Metrics copied from frozen evidence, not recalculated. Prediction row IDs/targets reconcile with prepared test records. | High for saved traceability, not independent metric reproduction |
| 5 | Validation, temporal splitting, leakage and limitations | PARTIALLY VERIFIED | `rasff_core.py:87–112,171–195,208–240`; Notebook 01 cell 12; `split_manifest.json`; Notebook 05 cells 9–23. Fitting boundaries are explicit; snapshot field availability, related-event leakage and historical analyst/test exposure cannot be ruled out. | High for code boundaries; limited for prospective validity |
| 6 | Notebook–bundle–Streamlit consistency | PARTIALLY VERIFIED | Notebook 03 cells 4, 10; Notebook 04 cells 4, 7; `app.py:18–19,59–86`; `artifacts/verification.json`; `verify_rebuild.py:30–54`. Shared prediction path and saved parity passes are present. Notebook 02 cell 10 has stale output; independent bundle/sample parity passed, but the fresh AppTest timed out (see final QA). | High for shared source path; saved-output caveat |
| 7 | Reproducibility, accessibility and portfolio presentation | PARTIALLY VERIFIED | `requirements.txt`, `tested_environment.json`, `artifacts/notebook_execution.json`, `README.md`, `RUN_GUIDE_KO.md`. Shipped artifacts support app-only use; clean-room installation, full notebook reproducibility, live deployment and accessibility have not been established. | Moderate |

## Claim-by-claim evidence

### Historical records, cleaning and target

`data_audit.json` reports **27,397 loaded and retained records**, zero rejected clean rows, 15 malformed raw records and 7,156 missing original hazards. Independent read-only CSV inspection confirms 27,397 rows and 27,397 unique references in both clean and prepared files, and zero prepared target mismatches. The source SHA-256 matches the audit and test report:

`108ee0009af0d194eb1609faf46b195eb9b2d7b3104a8652071b464d72f8c04a`

`prepare_data` normalizes strings/dates, rejects invalid dates/unknown labels/blank references, raises on duplicate references, sorts by date/reference and logs malformed raw records. These checks do **not** reconstruct upstream cleaning. The saved audit explicitly inherits manual correction 2021.2621 and previous exclusions. Its absolute source path is historical metadata, not a required runtime path.

Recorded label counts in `data_audit.json` and Notebook 01 cell 6: serious 14,756; not serious 4,134; undecided 3,029; potential risk 2,689; potentially serious 2,372; no risk 417. Target 1 means `serious`; target 0 includes all other five labels, including uncertain and potentially serious events. All supplied types are included: food, feed, food contact material and smaller other groups. The project is not a food-only or population-representative sample.

### Temporal separation and preprocessing

`temporal_blocks` partitions **unique observed calendar days**, using cutpoints at 60%, 75% and 85%; these are not row fractions or percentages of every elapsed calendar day. The saved manifest and independent prepared-row membership checks agree:

| Block | Records | First day | Last day | Use |
|---|---:|---|---|---|
| Train | 14,668 | 2020-01-02 | 2023-05-17 | Three expanding forward-CV folds; candidate fitting |
| Validation | 4,748 | 2023-05-18 | 2024-04-18 | Diagnostic comparison; later included in final development refit |
| Calibration | 3,128 | 2024-04-19 | 2024-11-25 | Sigmoid calibration only |
| Test | 4,853 | 2024-11-26 | 2025-10-31 | Frozen final evaluation |

`make_pipeline` fits subject/hazard TF-IDF, origin tokens and categorical one-hot encoding inside each fresh sklearn pipeline. There is no primary target encoding or learned full-snapshot preprocessing. One-hot encoding ignores unknown categories. `select_model` uses only training forward-CV mean macro F1, then AP as a tie-breaker; validation is not the ranking criterion. The main pipeline refits on train + validation (19,416 records). A separate Logistic Regression sigmoid calibrator (`C=1`, `lbfgs`) fits the logit of pipeline probabilities on calibration records only. Threshold is 0.50.

Verified code isolation is narrower than a universal “leakage-safe” claim: public fields may have been revised after the notification date; subject/hazards can encode decision-related information; related incidents across dates are not grouped; Notebook 01 cell 10 explores the full snapshot. The code guards an existing test report unless forced, but this is not an immutable or independently time-stamped proof of a single historical test access. No new performance tuning is justified by these saved results.

### Six fields and selected model

`feature_columns('raw')`, Notebook 03 cell 10 and the app's dynamic form contract identify:

`subject`, `origin`, `category`, `type`, `notifying_country`, `hazards`.

`classification`, follow-up fields, `Hazard_Type` and `simplified_hazard` are excluded from the primary model. `selected_spec.json` identifies **`raw_lr_C2`**, Logistic Regression with `C=2`, `liblinear`, `max_iter=500`, `random_state=42`. The saved candidate table ranks it highest among eligible primary candidates (CV macro F1 0.7964980126623017). `ablation_derived_lr` has a slightly higher saved CV score but is explicitly excluded from primary selection by the diagnostic-candidate rule. This distinction is consistent with Notebook 02 cell 8; it is not an unexplained selection mismatch.

### Frozen final metrics

Source: `artifacts/test_report.json`, `calibrated` object; matching rounded output in Notebook 03 cell 4. Final N = **4,853**, date range **2024-11-26 to 2025-10-31**.

| Metric | Saved value | Portfolio display |
|---|---:|---:|
| Accuracy | 0.7650937564393159 | 0.765 |
| Macro F1 | 0.7649171918660385 | 0.765 |
| Serious precision | 0.7217741935483871 | 0.722 |
| Serious recall | 0.7991071428571429 | 0.799 |
| Average precision | 0.8032597855603466 | 0.803 |
| ROC-AUC | 0.8440637642829808 | 0.844 |
| Brier score | 0.16229871916352137 | 0.162 |
| Log loss | 0.49581424042786576 | 0.496 |

Saved confusion matrix, actual/predicted order `[0,1]`: `[[1923,690],[450,1790]]`. The report also separates uncalibrated, prior-baseline and development-training scores. Do not confuse these with calibrated final test scores. No confidence intervals or external validation are supplied. Metrics were **not recalculated** in this audit.

### Classification late-information ablation

Notebook 05 cells 5, 7, 9, 11 and 13 inspect the saved contract, clone its pipeline and fit two fresh models on the same training block. Scenario B adds only `classification` using a clone of the categorical transformer. Both evaluate the same 4,748 validation records at threshold 0.5, **without final sigmoid calibration**. Saved cell 15 reports:

| Validation diagnostic | Scenario A: six fields | Scenario B: +classification |
|---|---:|---:|
| Accuracy | 0.770008 | 0.842460 |
| Macro F1 | 0.769836 | 0.842444 |
| ROC-AUC | 0.846909 | 0.913537 |

Cell 17 matches Scenario A against the earlier saved validation scores; cell 19 reports macro-F1 delta **0.072608**. Cell 21 reports 8,220 vs 8,225 transformed dimensions. Cell 23 has `RUN_POSTHOC_TEST=False` and output “Post-hoc final-test comparison not run.” This is evidence of added retrospective signal, not timely operational availability or a replacement model. No ablation rerun or separate new artifact was produced here. Cell 6's wording “final test-fitted model” is inaccurate: the primary pipeline is fitted on train + validation, not test; this markdown defect is recorded without editing the notebook.

### Bundle and app parity

`fit_final` serializes the pipeline, calibrator, columns, threshold, target/scope, source hash and versions to `artifacts/final_bundle.joblib`. `predict_bundle` applies normalization, selected columns, pipeline probabilities and the calibrator. Both Notebook 03 and `app.py` invoke this function. `app.py` loads the shipped model and does not train; Explorer filtering does not modify its prediction model.

`artifacts/verification.json` records passes for date separation, unseen token/category handling, default feature scope, save/reload parity, app filtering and prediction parity. Inspection of `verify_rebuild.py` confirms meaningful assertions, but the script also fits a model and writes artifacts, so it was **not run** during this audit. These saved results are historical evidence, not fresh test results. `notebook_execution.json` covers notebooks 01–04 only and records in-process IPython execution, not a separate Jupyter kernel run. Notebook 04's optional UI smoke cell is disabled; the saved UI verification comes from the verifier.

## Issue register

| ID | Priority | Finding / evidence | Disposition |
|---|---|---|---|
| P1-01 | P1 | Inherited source cleaning and correction 2021.2621 not independently reconciled; `data_audit.json`, Notebook 01 cell 3. | Open scientific limitation. Document prominently; no data changes authorized. Does not by itself invalidate retrospective snapshot metrics. |
| P1-02 | P1 | Prediction-time availability and incident independence unverified; full-snapshot EDA visible. Temporal splitting does not establish prospective validity or eliminate every leakage route. | Open scientific limitation. Restrict claims to retrospective recorded-label classification. Prospective use needs a new design and untouched evaluation data. |
| P2-01 | P2 | Notebook 02 cell 10 saved table contains five primary candidates while cell 8 and CSV artifacts contain seven including ablations. Execution counts 1,7,8,9,6 show mixed output history. | Open saved-output inconsistency; primary selection/metrics agree. Cite authoritative artifacts, avoid claiming every notebook output is synchronized. No rerun permitted. |
| P2-02 | P2 | Rebuild quick start implies notebooks must run before app; broad “all notebook code cells” claim exceeds execution manifest coverage (01–04). | Correct in documentation: app-only launch uses shipped files; distinguish recorded execution from fresh reproduction. |
| P2-03 | P2 | No verified public Rebuild deployment URL in root/Rebuild documentation; no clean-room install or accessibility assessment. | Local app entry point only; do not invent a live link. Deployment remains unverified. |
| P2-04 | P2 | Notebook 05 cell 6 says “final test-fitted model” despite development-only fitting code. | Document correct interpretation here; notebook preserved under minimal-documentation scope. |
| P2-05 | P2 | No tracked license file found. Public repository availability alone does not establish reuse terms. | Owner decision remains open; do not invent a license or upstream dataset permission. |

**P0:** none identified within this evidence-based scope. Confidence is high for saved artifact identity and implementation boundaries, moderate for end-to-end historical execution, and limited for upstream provenance, field timing and deployment. Absence of an identified P0 is not proof that all scientific risks have been excluded.

## QA and artifact integrity (initial audit, with final update below)

### PASS — checks performed in this session

- Initial clean branch status and restricted scope inspection.
- AST syntax parsing of three Rebuild Python files and all code cells in six notebooks, without execution; no saved notebook error outputs found.
- Read-only CSV row/reference checks, target construction, all four manifest membership counts and 4,853 test prediction reference/target alignment.
- SHA-256 verification of the clean source against saved audit/test-report metadata.
- Static trace of model selection, fitting/calibration boundaries, ablation and shared app prediction path.

### FAIL

- Fresh Streamlit AppTest did not complete: the first combined attempt stalled and was interrupted; the bounded app-only attempt printed its start message but exceeded a 50-second watchdog before a render pass. No application exception or root cause was established. This is a failed smoke-test attempt, not evidence that the published metrics are invalid.
- Inspection scripts initially encountered Windows console/default-file encoding errors. Explicit UTF-8 retries passed; these were audit-tool issues.

### NOT RUN

- Training, notebook reruns, metric recalculation and the training-containing `verify_rebuild.py`: intentionally not run.
- Clean-room dependency installation, public deployment/browser verification, accessibility testing and prospective/external validation: not run.
- Fresh UI filter/form assertions: not reached because AppTest rendering timed out. Historical filter/form parity results remain saved evidence only.

Evidence hashes at audit baseline:

| File | SHA-256 |
|---|---|
| `data/prepared.csv` | `fc9eacd7ed73a9e2f9862b74dd554fa335c1c5908395a7f53869f110da6fa586` |
| `artifacts/final_bundle.joblib` | `6c9f544e4514f85cdbdb7f472ffa97fdc66f202718b4b6ac871010a8072c9b27` |
| `artifacts/test_report.json` | `19557956eb0e7c683fb0f018b1a3325775cbba15514c080cf779448ff94822d5` |
| `artifacts/test_predictions.csv` | `ef9708ee72c2adefc954e3d37c6c667579aa8bc31d83c2f62788d14376a18274` |

## Release assessment and unresolved decisions

Portfolio readiness is limited to a transparent historical learning project. The README should lead with the current app, question, saved calibrated metrics and methodological contribution, identify Rebuild as the sole current implementation, and briefly mark other folders as archives. Do not compare the historical implementation's scores or claim regulatory fitness.

Unresolved owner decisions: reconcile original cleaning against source records; establish field timing/related-event handling before any prospective work; publish and verify a Rebuild app URL if desired; choose licensing terms; decide whether to refresh stale notebook outputs in a separately authorized execution cycle. None requires modifying the frozen model or delaying an explicitly caveated documentation release.


## Final QA and readiness update

Completed after the initial report and README corrections, before commit:

| Check | Outcome | Evidence / qualification |
|---|---|---|
| Bundle identity and contract | PASS | Loaded `final_bundle.joblib`; `spec` equals `selected_spec.json`, classifier C=2, six columns in expected order, threshold 0.5, source hash equals `data_audit.json`. Bundle records sklearn 1.8.0 and Python 3.12.14. |
| Saved prediction and reload parity | PASS | Read prepared records matching the first 20 saved test references; `predict_bundle` matches saved `p_serious` at rtol=1e-10 / atol=1e-12; a separately reloaded bundle matches. Inference only; no performance metrics recomputed. This samples 20 records, not all test predictions. |
| Fresh Streamlit AppTest rendering | FAIL | Bounded retry exceeded 50 seconds before a render pass; app test used `default_timeout=15`. Training methods and artifact JSON writing were patched to raise within the test process. No scientific/source files were edited. Cause unresolved. |
| Fresh UI filter and form parity | NOT RUN | Assertions not reached after render timeout. Do not relabel historical `verification.json` results as fresh passes. |
| Internal Markdown links | PASS | Relative link targets in both READMEs and this report exist. Public app endpoint not verified; no live app link added. External project URLs were not all health-checked. |
| README figures | PASS | Eight displayed metrics match three-decimal formatting of the frozen calibrated report; test N matches. Snapshot/split counts also reconciled independently. |
| Syntax/static checks | PASS | AST parsing for three Python files and every code cell in six notebooks; no execution. `git diff --check` passes. |
| Artifact preservation | PASS | Key hashes above unchanged; tracked diff contains only the two READMEs, plus this newly added report. No code, notebook, model, data or evaluation artifacts changed. |
| Full verifier, training, notebook execution, metric recalculation | NOT RUN | Excluded by requested scope; verifier contains `.fit` and artifact writes. |
| Clean-room install / public-browser test / accessibility | NOT RUN | No deployment or clean-room reproduction claim. |

Runtime used for lightweight checks: Python 3.13.5, pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.8.0, Streamlit 1.64.0, joblib 1.5.3. This differs from the recorded training environment. pandas emitted an optional `numexpr` version warning; independent bundle inference nevertheless passed. Streamlit emitted a bare-mode ScriptRunContext warning before the timeout. These observations do not identify the timeout cause.

**Final readiness: READY WITH CAVEATS.** Documentation corrections resolve P2-02 and make P2-03 explicit; scientific P1 limitations remain. Criterion 6 remains PARTIALLY VERIFIED because the source/bundle sample agreement and saved UI passes coexist with stale notebook output and an unsuccessful fresh UI check. Add the UI timeout to P2-03 as an unresolved local-runtime verification issue. No material P0 was found, so the documentation-only release is eligible for push. A future successful app test in the pinned environment is needed before claiming fresh end-to-end UI verification.

Release scope: root `README.md`, `RASFF_Rebuild/README.md`, and this report only. Commit identity is supplied in the session release summary rather than embedding a self-referential commit SHA here.
