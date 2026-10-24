"""Empirical upstream-score error on the public MIMIC-IV-ED demo (v1.3.0).

A simple rule-based upstream score (partial NEWS2 from triage vital signs:
respiratory rate, SpO2, systolic BP, heart rate, temperature; consciousness and
supplemental oxygen are not recorded at triage and are omitted) is mapped to an
ORASR risk score and compared with the triage nurse's Emergency Severity Index
(ESI) acuity, used here as the reference for "high risk" (ESI 1-2). The result
is an *empirical* error distribution of a plausible upstream score, not a
clinical validation of either score.

Mapping (transparent):
    NEWS2_partial >= 7                      -> rho = 0.85 (Safe band)
    5-6, or any single parameter scoring 3 -> rho = 0.50 (Normal band)
    otherwise                               -> rho = 0.15 (Fast band)
    ESI 1 -> 0.95, 2 -> 0.80, 3 -> 0.50, 4 -> 0.20, 5 -> 0.10 (reference score)

Usage:  python -I scripts/ed_acuity_disagreement.py --ed-path D:/AVS/Datasets/mimic-ed-demo
Output: results/ed_acuity_disagreement.json (aggregate counts only)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ESI_RHO = {1: 0.95, 2: 0.80, 3: 0.50, 4: 0.20, 5: 0.10}


def _band(x: float, cuts) -> int:
    for upper, pts in cuts:
        if x <= upper:
            return pts
    return cuts[-1][1]


def news2_partial(rr: float, spo2: float, sbp: float, hr: float, temp_c: float):
    parts = {
        "rr": _band(rr, [(8, 3), (11, 1), (20, 0), (24, 2), (1e9, 3)]),
        "spo2": _band(spo2, [(91, 3), (93, 2), (95, 1), (1e9, 0)]),
        "sbp": _band(sbp, [(90, 3), (100, 2), (110, 1), (219, 0), (1e9, 3)]),
        "hr": _band(hr, [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (1e9, 3)]),
        "temp": _band(temp_c, [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (1e9, 2)]),
    }
    return sum(parts.values()), max(parts.values())


def rho_from_news(total: int, max_single: int) -> float:
    if total >= 7:
        return 0.85
    if total >= 5 or max_single == 3:
        return 0.50
    return 0.15


def fnum(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def band_name(rho: float) -> str:
    return "FAST" if rho < 0.3 else ("NORMAL" if rho < 0.7 else "SAFE")


def run(ed_root: Path) -> Dict:
    path = ed_root / "ed" / "triage.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n_all = len(rows)
    usable, residuals = [], []
    excluded = {"missing_acuity": 0, "missing_vitals": 0, "implausible_vitals": 0}
    for r in rows:
        esi = fnum(r.get("acuity"))
        vit = [fnum(r.get(k)) for k in ("resprate", "o2sat", "sbp", "heartrate", "temperature")]
        if esi is None:
            excluded["missing_acuity"] += 1
            continue
        if any(v is None for v in vit):
            excluded["missing_vitals"] += 1
            continue
        rr, spo2, sbp, hr, temp_f = vit
        temp_c = (temp_f - 32.0) * 5.0 / 9.0
        if not (2 <= rr <= 80 and 50 <= spo2 <= 100 and 40 <= sbp <= 300 and 20 <= hr <= 250
                and 30 <= temp_c <= 44):
            excluded["implausible_vitals"] += 1
            continue
        total, mx = news2_partial(rr, spo2, sbp, hr, temp_c)
        rho_rule = rho_from_news(total, mx)
        rho_ref = ESI_RHO[int(esi)]
        usable.append((int(esi), total, rho_rule, rho_ref))
        residuals.append(round(rho_rule - rho_ref, 4))
    conf: Dict[str, Dict[str, int]] = {}
    for _, _, rr_, ref in usable:
        conf.setdefault(band_name(ref), {}).setdefault(band_name(rr_), 0)
        conf[band_name(ref)][band_name(rr_)] += 1
    high = [u for u in usable if u[0] <= 2]
    high_down = sum(1 for u in high if u[2] < 0.7)
    high_fast = sum(1 for u in high if u[2] < 0.3)
    low = [u for u in usable if u[0] >= 3]
    low_up = sum(1 for u in low if u[2] >= 0.7)
    res_counts: Dict[str, int] = {}
    for r in residuals:
        res_counts[f"{r:+.2f}"] = res_counts.get(f"{r:+.2f}", 0) + 1
    return {
        "source": "MIMIC-IV-ED demo (public, open access) ed/triage.csv.gz",
        "n_triage_rows": n_all, "excluded": excluded, "n_usable": len(usable),
        "esi_counts": {str(k): sum(1 for u in usable if u[0] == k) for k in range(1, 6)},
        "confusion_reference_band_vs_rule_band": conf,
        "esi_1_2": {"n": len(high), "rule_below_safe": high_down,
                    "rule_below_safe_pct": round(100 * high_down / len(high), 2) if high else None,
                    "rule_fast": high_fast},
        "esi_3_5": {"n": len(low), "rule_safe": low_up,
                    "rule_safe_pct": round(100 * low_up / len(low), 2) if low else None},
        "residual_rule_minus_reference": {"mean": round(sum(residuals) / len(residuals), 4),
                                          "counts": res_counts},
        "mapping": {"news2": ">=7 -> 0.85; 5-6 or any single 3 -> 0.50; else 0.15",
                    "esi_reference": ESI_RHO},
        "caveat": ("Partial NEWS2 (no consciousness or supplemental-oxygen item); ESI is a nurse "
                   "triage judgement, not an outcome. The residual distribution is used only as an "
                   "empirical error model for routing, not as evidence about either score."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-path", default=os.environ.get("MIMIC_ED_PATH", "D:/AVS/Datasets/mimic-ed-demo"))
    ap.add_argument("--out", default=str(RESULTS / "ed_acuity_disagreement.json"))
    args = ap.parse_args()
    root = Path(args.ed_path)
    if not (root / "ed" / "triage.csv.gz").exists():
        sys.stderr.write(f"MIMIC-IV-ED demo not found under {root}; nothing written.\n")
        sys.exit(2)
    out = run(root)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print({k: out[k] for k in ("n_usable", "esi_counts", "esi_1_2", "esi_3_5")})
    print("confusion", out["confusion_reference_band_vs_rule_band"])


if __name__ == "__main__":
    main()
