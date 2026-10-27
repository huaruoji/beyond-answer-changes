"""Summarise a built CTL set: realised axes, gate status, and the power implied.

Answers the exploration objective's last question — **how many items are needed** —
from measured variance rather than from an assumed effect size. Reports the per-item
distribution of the two contrast magnitudes, because a median gap says nothing about
how many items will actually carry a measurable manipulation.

Usage::

    python scripts/ctl_power_summary.py --dir outputs/ctl/sample30

Reads ``build_audit.json`` (per-cell axes) and writes ``power_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MASKER_PAIR = ("low_pass_400", "event_matched")


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def n_for_effect(sd: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items per arm for a two-sided two-sample comparison of a continuous measure."""

    z_alpha = normal_quantile(1 - alpha / 2)
    z_beta = normal_quantile(power)
    return int(np.ceil(2 * ((z_alpha + z_beta) * sd / effect) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--target-accuracy-gap", type=float, default=0.10,
                        help="accuracy difference (proportion) the pilot should be able to resolve")
    args = parser.parse_args()

    audit = json.loads((args.dir / "build_audit.json").read_text(encoding="utf-8"))
    items = audit["items"]
    a, b = MASKER_PAIR

    ee_gap, ea_mismatch, ea_energy_gap, realised_drop, baselines = [], [], [], [], []
    per_class: dict[str, list[float]] = {}
    for entry in items:
        cells = entry["cells"]
        va = cells[f"equal_energy__{a}"]["audibility_db"]
        vb = cells[f"equal_energy__{b}"]["audibility_db"]
        sa = cells[f"equal_energy__{a}"]["energy_snr_db"]
        sb = cells[f"equal_energy__{b}"]["energy_snr_db"]
        ma = cells[f"equal_audibility__{a}"]["audibility_db"]
        mb = cells[f"equal_audibility__{b}"]["audibility_db"]
        na = cells[f"equal_audibility__{a}"]["energy_snr_db"]
        nb = cells[f"equal_audibility__{b}"]["energy_snr_db"]
        ee_gap.append(abs(va - vb))
        ea_mismatch.append(abs(ma - mb))
        ea_energy_gap.append(abs(na - nb))
        realised_drop.append(entry.get("margin_drop_realised_db", float("nan")))
        baselines.append(entry.get("baseline_margin_db", float("nan")))
        per_class.setdefault(entry["class"], []).append(abs(va - vb))

    ee = np.asarray(ee_gap, dtype=float)
    ea = np.asarray(ea_mismatch, dtype=float)
    eg = np.asarray(ea_energy_gap, dtype=float)

    # how many items carry a usable manipulation (>= 10 dB audibility separation)
    usable = int((ee >= 10.0).sum())
    # accuracy-based power: the pilot's endpoint is a difference in MIG, a proportion.
    # SE of a difference of two paired proportions at p~0.5 is sqrt(2*0.25/n); solve for n.
    p = 0.5
    se_needed = args.target_accuracy_gap / (normal_quantile(0.975) + normal_quantile(0.8))
    n_accuracy = int(np.ceil(2 * p * (1 - p) / se_needed ** 2))

    summary = {
        "n_items": len(items),
        "masker_pair": list(MASKER_PAIR),
        "equal_energy_audibility_gap_db": {
            "median": float(np.median(ee)), "mean": float(ee.mean()), "sd": float(ee.std(ddof=1)),
            "min": float(ee.min()), "max": float(ee.max()),
            "p10": float(np.percentile(ee, 10)), "p90": float(np.percentile(ee, 90)),
            "n_at_least_10db": usable,
            "fraction_usable": usable / len(ee),
        },
        "equal_energy_energy_gap_db": {
            "median": float(np.median([abs(cells_gap) for cells_gap in
                [abs(e["cells"][f"equal_energy__{a}"]["energy_snr_db"] - e["cells"][f"equal_energy__{b}"]["energy_snr_db"]) for e in items]])),
        },
        "equal_audibility_mismatch_db": {
            "median": float(np.median(ea)), "max": float(ea.max()), "sd": float(ea.std(ddof=1)),
        },
        "equal_audibility_energy_gap_db": {
            "median": float(np.median(eg)), "sd": float(eg.std(ddof=1)), "min": float(eg.min()),
        },
        "realised_margin_drop_db": {
            "median": float(np.nanmedian(realised_drop)),
            "min": float(np.nanmin(realised_drop)), "max": float(np.nanmax(realised_drop)),
        },
        "baseline_margin_db": {
            "median": float(np.nanmedian(baselines)),
            "min": float(np.nanmin(baselines)), "max": float(np.nanmax(baselines)),
        },
        "gate5_equal_energy_separates": bool(np.median(ee) >= 10.0),
        "gate6_equal_audibility_matches": bool(np.median(ea) <= 1.0),
        "power": {
            "target_accuracy_gap": args.target_accuracy_gap,
            "n_items_per_cell_for_accuracy_gap": n_accuracy,
            "n_for_10db_audibility_effect": n_for_effect(float(ee.std(ddof=1)), 10.0),
            "n_for_5db_audibility_effect": n_for_effect(float(ee.std(ddof=1)), 5.0),
            "note": "accuracy-based n is a binomial approximation; the audibility-based n "
                    "treats the gap itself as the endpoint, which is how the pilot should be read",
        },
        "per_class_median_gap_db": {k: float(np.median(v)) for k, v in sorted(per_class.items())},
    }

    out = args.dir / "power_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"items                                   : {summary['n_items']}")
    print(f"equal-energy audibility gap  median/sd  : {summary['equal_energy_audibility_gap_db']['median']:.2f} / "
          f"{summary['equal_energy_audibility_gap_db']['sd']:.2f} dB   "
          f"[p10 {summary['equal_energy_audibility_gap_db']['p10']:.2f}, p90 {summary['equal_energy_audibility_gap_db']['p90']:.2f}]")
    print(f"  items with >= 10 dB separation        : {usable}/{len(ee)} ({usable/len(ee):.0%})")
    print(f"equal-audibility margin mismatch median : {summary['equal_audibility_mismatch_db']['median']:.3f} dB "
          f"(max {summary['equal_audibility_mismatch_db']['max']:.3f})")
    print(f"equal-audibility energy gap median      : {summary['equal_audibility_energy_gap_db']['median']:.2f} dB")
    print(f"realised margin drop median [min, max]  : {summary['realised_margin_drop_db']['median']:.1f} "
          f"[{summary['realised_margin_drop_db']['min']:.1f}, {summary['realised_margin_drop_db']['max']:.1f}] dB")
    print(f"baseline margin (no masker) min..max    : {summary['baseline_margin_db']['min']:.1f} .. "
          f"{summary['baseline_margin_db']['max']:.1f} dB")
    print(f"gates 5 / 6                             : {summary['gate5_equal_energy_separates']} / "
          f"{summary['gate6_equal_audibility_matches']}")
    pw = summary["power"]
    print(f"n per cell for a {pw['target_accuracy_gap']:.0%} accuracy gap    : {pw['n_items_per_cell_for_accuracy_gap']}")
    print(f"n for a 10 dB audibility effect         : {pw['n_for_10db_audibility_effect']}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
