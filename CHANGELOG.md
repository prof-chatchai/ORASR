# Changelog

All notable changes to this repository are documented here.

## 1.3.0 (2026-10-24) - resubmission analyses

Library
- Fail-closed input handling: missing, non-numeric, Boolean, NaN, infinite or out-of-range
  risk scores (and invalid uncertainties) route SAFE and are blocked (`R0_RiskInput`).
  Behaviour change: negative scores no longer route FAST.
- Optional uncertainty-aware routing (`uncertainty_k`, `risk_uncertainty=`), action-type
  floor (`high_alert_types`), signed approval tokens (`orasr.approval`), two-phase and
  compensable execution (`orasr.staging`), hash-chained JSONL audit log (`orasr.audit`).
  All opt-in; defaults reproduce v1.2.0 routing for every valid score.

Analyses (scripts -> results)
- clinical_config.py / detection_clinical.py -> detection_clinical.json
- mimic_routing.py -> mimic_demo_routing.json (public MIMIC-IV demo; mimic_full_routing.json
  when run on the credentialed database, with small-cell suppression)
- ed_acuity_disagreement.py -> ed_acuity_disagreement.json (MIMIC-IV-ED demo)
- structured_misrouting.py -> structured_misrouting.json, mitigation_comparison.json
- upstream_reliability.py, baselines_extended.py, threshold_derivation.py,
  latency_realistic.py, assurance_mechanisms.py -> results of the same names
- misrouting_analysis.py clips noisy scores to [0, 1] (unchanged results).
- Found and documented: the v1.2.0 RISK_RULES mapping in extract_mimic.py routes no order
  FAST on real data (substring matching); kept unchanged for comparison, superseded by
  the word-boundary mapping v2 in mimic_routing.py.

## Unreleased

- Standardized root metadata, authorship, reference, testing, and container files.
- Verified local tests and cleaned generated cache or coverage artifacts before publishing.
