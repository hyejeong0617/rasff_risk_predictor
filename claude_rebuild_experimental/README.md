# Claude-assisted rebuild — experimental archive

This folder preserves the **intermediate AI-assisted rebuild** that previously lived at the repository root.

It is retained because it contains useful ideas and documents an important stage in the project history, including:

- chronological evaluation concepts
- richer feature engineering
- missingness analysis
- sentence embeddings
- target encoding
- Optuna/XGBoost experimentation
- a three-page Streamlit interface

## Why this is not the current portfolio baseline

A later audit found that the project mixed useful engineering ideas with assumptions and implementation choices that were not sufficiently validated end-to-end.

In particular:

- the public dataset does not by itself prove the exact availability time of every field;
- some README language framed the task more strongly as pre-decision prediction than the evidence supported;
- the Streamlit predictor used a separate heuristic scoring path rather than the trained model described in the modeling workflow;
- the feature/preprocessing stack became difficult to defend as one reproducible prediction contract.

For those reasons, this version is **preserved, not deleted**, but it is no longer presented as the final implementation.

## Use this folder for

- understanding the evolution of the project;
- reviewing ideas that may deserve a new, separately validated experiment;
- comparing a complex AI-assisted rebuild with the later smaller audited pipeline.

## Do not use this folder for

- quoting the repository's current final model performance;
- claiming a validated pre-decision RASFF risk predictor;
- treating the old Streamlit probability as output from the trained XGBoost model.

For the current audited version, go to [`../RASFF_Rebuild/`](../RASFF_Rebuild/).

The original README from this experimental stage is preserved as [`README_ORIGINAL.md`](./README_ORIGINAL.md) so that the historical claims remain inspectable rather than silently rewritten.
