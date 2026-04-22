# Misinformation Risk Triage Analyzer

This project builds and evaluates **machine learning systems for misinformation risk triage**.

The focus is not "automatic truth detection." The focus is:
- ranking content by risk,
- explaining what signals drive model behavior,
- supporting human analysts with transparent evidence.

## What problem this project solves
Large content streams cannot all be reviewed manually. A triage model can prioritize likely high-risk items for human review, reducing analyst load while preserving oversight.

## Project maturity (honest status)
- Current status: **research + prototype**
- Production readiness: **in progress**
- Most important known gap: limited diffusion/network features and limited cross-domain robustness

## Repository Structure
- `app/` Streamlit interface and project narrative
- `src/` data processing and training/evaluation scripts
- `notebooks/` experiment notebooks and exploratory work
- `outputs/` evaluation artifacts and figures
- `models/` serialized models and manifests
- `PROJECT_UPGRADE_PLAN.md` professionalization roadmap

## Key Experiments
1. Metadata-only baseline (COVID graph-derived metadata)
2. Text-only baseline (ISOT and FakeNewsNet PolitiFact)
3. Hybrid models (text + metadata)
4. Controlled Stage 3B comparison (same dataset for fair text/meta/hybrid comparison)

## Professionalization Upgrade Added
A new training/evaluation entrypoint is included:
- `src/train_eval_politifact_tfidf.py`

It introduces:
- train/validation/test protocol
- threshold selection on validation only
- group-aware splitting by source domain (fallback to stratified split)
- saved artifact manifest and evaluation outputs

Run it with:

```bash
python3 src/train_eval_politifact_tfidf.py
```

Outputs:
- `models/tfidf_model_pro.joblib`
- `models/lr_model_pro.joblib`
- `models/model_manifest_pro.json`
- `outputs/professional_eval/validation_threshold_table.csv`
- `outputs/professional_eval/test_metrics.json`

## Current Interpretation of Results
- Text is currently the strongest standalone signal.
- Lightweight metadata adds complementary but limited value.
- This is useful for **risk triage**, not for autonomous fact verification.

## Responsible Use
This repository should be used for research/education and analyst-assistive tooling.
It should not be used as a sole decision-maker for content moderation, sanctions, or claims of factual truth.

## Next Steps
See `PROJECT_UPGRADE_PLAN.md` for prioritized upgrades across evaluation rigor, model depth, reliability, and deployment safety.
