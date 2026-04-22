# Project Upgrade Plan (Professionalization Roadmap)

This roadmap turns the repository from an academic prototype into a credible misinformation risk-triage ML project.

## Why this matters
- The current project is strong for controlled experimentation.
- Real-world use requires leakage-safe evaluation, reproducible pipelines, and explicit risk controls.
- Professional hiring signal comes from reliability and rigor, not only model accuracy.

## Phase 1 (Completed in this pass)
- Added `src/train_eval_politifact_tfidf.py` with:
  - train/validation/test protocol
  - validation-only threshold selection
  - group-aware splitting by source domain (fallback to stratified when needed)
  - artifact + metrics manifest output
- Added test coverage for core utility logic.
- Reframed README around "risk triage" and deployment-aware scope.

## Phase 2 (Next highest impact)
- Replace notebook-only result generation with script-driven experiment runs.
- Add calibration evaluation (Brier score, reliability curve).
- Add domain-shift checks (train on one source mix, test on another).
- Add robust error handling and schema checks for all raw datasets.

## Phase 3 (Modeling depth)
- Add richer metadata features:
  - propagation velocity and burstiness
  - simple graph/cascade shape proxies
  - account-level aggregate credibility proxies where available
- Add transformer baseline (DistilBERT/DeBERTa) under the same split policy.
- Add slice metrics (long vs short text, source domain buckets).

## Phase 4 (Production quality)
- CI: run tests and sanity checks on every PR.
- Add `requirements-dev.txt` and pinned runtime requirements.
- Add model card and dataset card (intended use, risks, failure modes).
- Add inference API entrypoint with structured JSON output and model version metadata.

## Phase 5 (Real-world framing)
- Position product as analyst-assistive triage, not truth arbiter.
- Add a human-in-the-loop workflow spec:
  - model flags content
  - analyst review required
  - decision audit log with reason codes
- Add policy guardrails in app messaging to prevent misuse.
