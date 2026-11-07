"""Recompute the paper's reported numbers from the result files shipped in this repository and check them.

Needs only numpy and scipy: no videos, point clouds or CT volumes. Each line prints the recomputed value next to the
value in the manuscript tables; the script exits non-zero if any check fails.

    python check_paper_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
REP = ROOT / "runs/own_data/reports"
load = lambda p: json.load(open(ROOT / p))
fails = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {label:58s} {got:9.3f}   (paper {want})")
    if not ok: fails.append(label)


def stats(rec, ct):
    rec, ct = np.asarray(rec, float), np.asarray(ct, float); e = rec - ct
    sd = e.std(ddof=1)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), n=len(e),
                loa=(e.mean() - 1.96 * sd, e.mean() + 1.96 * sd), r=pearsonr(rec, ct)[0], rho=spearmanr(rec, ct)[0])


print("Table: CT accuracy, all-slice baseline (ct_validated_3/CT_validation_metrics.json)")
base = load("runs/own_data/ct_validated_3/CT_validation_metrics.json")
paper_base = {"2-V2": (1.44, -0.01, 27), "20-V1": (1.13, +0.16, 31), "50-V2": (2.41, +0.92, 44)}
R, C = [], []
for case, (rm, bi, n) in paper_base.items():
    s = stats(base[case]["recon_dce"], base[case]["ct_dce"]); R += base[case]["recon_dce"]; C += base[case]["ct_dce"]
    check(f"{case} baseline RMSE (mm)", s["rmse"], rm, 0.005); check(f"{case} baseline bias (mm)", s["bias"], bi, 0.005)
    check(f"{case} baseline n", s["n"], n, 0)
s = stats(R, C)
check("baseline pooled RMSE (mm)", s["rmse"], 1.85, 0.005); check("baseline pooled bias (mm)", s["bias"], 0.45, 0.005)
check("baseline pooled n", s["n"], 102, 0); check("baseline pooled Pearson r", s["r"], 0.82, 0.005)
check("baseline LoA lower (mm)", s["loa"][0], -3.10, 0.01); check("baseline LoA upper (mm)", s["loa"][1], 3.99, 0.01)
check("baseline mean of patients (mm)", np.mean([stats(base[c]["recon_dce"], base[c]["ct_dce"])["rmse"] for c in base]), 1.66, 0.005)

print("\nTable: CT accuracy, gated estimator (adopted rows)")
gated = {"2-V2": load("runs/own_data/reports/rerun_eval_2-V2.json")["methods"]["M1 cam-centerline + cov>=0.75"],
         "20-V1": load("runs/own_data/reports/ct3_regate.json")["20-V1"]["circle-only"],
         "50-V2": load("runs/own_data/reports/rerun_eval_50-V2.json")["methods"]["M2 partial-arc gated"]}
paper_gated = {"2-V2": (1.07, -0.05, 40, 0.00), "20-V1": (0.83, +0.10, 24, -0.05), "50-V2": (1.00, +0.02, 16, 0.74)}
R, C, per = [], [], []
for case, (rm, bi, n, rho) in paper_gated.items():
    g = gated[case]; s = stats(g["dce"], g["ct"]); R += g["dce"]; C += g["ct"]; per.append(s["rmse"])
    check(f"{case} gated RMSE (mm)", s["rmse"], rm, 0.005); check(f"{case} gated bias (mm)", s["bias"], bi, 0.005)
    check(f"{case} gated n", s["n"], n, 0); check(f"{case} gated Spearman rho (S8)", s["rho"], rho, 0.005)
s = stats(R, C)
check("gated pooled RMSE (mm)", s["rmse"], 0.99, 0.005); check("gated pooled bias (mm)", s["bias"], 0.01, 0.005)
check("gated pooled n", s["n"], 80, 0); check("gated pooled Pearson r", s["r"], 0.95, 0.005)
check("gated LoA lower (mm)", s["loa"][0], -1.94, 0.01); check("gated LoA upper (mm)", s["loa"][1], 1.96, 0.01)
check("gated mean of patients (mm)", np.mean(per), 0.97, 0.005)

print("\nTable: ablation (rerun_eval_*.json; M0 = no gate, M1 = coverage gate, M2 = partial-arc)")
m = lambda f, k: load(f"runs/own_data/reports/{f}")["methods"][k]
M0, M1, M2 = "M0 cam-centerline, no gate", "M1 cam-centerline + cov>=0.75", "M2 partial-arc gated"
for f, k, rm, n, lab in [("rerun_eval_50-V2.json", M0, 2.16, 43, "50-V2 centred re-window"),
                         ("rerun_eval_50-V2.json", M1, 1.89, 34, "50-V2 + coverage gate"),
                         ("rerun_eval_50-V2.json", M2, 1.00, 16, "50-V2 + partial-arc fit"),
                         ("rerun_eval_2-V2_f2870-3370.json", M0, 1.28, 34, "2-V2 single withdrawal pass"),
                         ("rerun_eval_2-V2_f2870-3370.json", M2, 1.02, 20, "2-V2 single pass, partial-arc (rejected)"),
                         ("rerun_eval_2-V2.json", M1, 1.07, 40, "2-V2 loop model, withdrawal camera path"),
                         ("rerun_eval_2-V2_f880-2100.json", M1, 1.63, 32, "2-V2 loop model, descent camera path")]:
    check(f"{lab} RMSE (mm)", m(f, k)["rmse"], rm, 0.005); check(f"{lab} n", m(f, k)["n"], n, 0)
for f, want, lab in [("rerun_eval_50-V2.json", 1.18, "50-V2"), ("rerun_eval_2-V2_f2870-3370.json", 0.59, "2-V2 single pass"),
                     ("rerun_eval_2-V2.json", 0.57, "2-V2 loop")]:
    check(f"self-consistency ratio {lab} (M2/M1 calibre)", m(f, M2)["med_dce"] / m(f, M1)["med_dce"], want, 0.005)

print("\nS8: 20-V1 CT-reference sensitivity (ct_reference_sensitivity_20-V1.json)")
sens = load("runs/own_data/reports/ct_reference_sensitivity_20-V1.json")
check("TotalSegmentator reference, re-registered (mm)", sens["TotalSegmentator (published reference)"]["rmse"], 0.92, 0.005)
check("raw -900 HU, z 1190-1174 (mm)", sens["raw -900 HU, z 1190-1174 (17 levels)"]["rmse"], 0.75, 0.005)
check("raw -900 HU, z 1221-1206 (mm)", sens["raw -900 HU, z 1221-1206 (14 levels)"]["rmse"], 0.69, 0.005)

print("\nS3: 20-V1 CT extraction (rawlumen_*.json)")
ins, exp = load("runs/own_data/reports/rawlumen_INSP.json"), load("runs/own_data/reports/rawlumen_EXP.json")
for d in (ins, exp):
    print(f"  {d['label']}: -900 HU column z {d['z_first']:.1f} -> {d['z_last']:.1f} ({d['n_slices']} slices), carina z {d['carina_z']:.1f}")
check("carina craniocaudal shift EXP - INSP (mm)", exp["carina_z"] - ins["carina_z"], 14.0, 0.05)

print("\nYield / S10 (yield_table.json)")
y = load("runs/own_data/reports/yield_table.json")
tier = lambda *t: sum(r["tier"] in t for r in y)
check("examinations", len(y), 29, 0)
check("measurable (CT-validated + measurable tube)", tier("CT-validated", "measurable tube"), 18, 0)
check("failures (partially + not measurable)", tier("partially measurable", "not measurable"), 11, 0)
check("first-batch examinations", sum(r["batch"] == "first" for r in y), 8, 0)

print("\nS5: capture statistics (capture_table.json)")
for r in load("runs/own_data/reports/capture_table.json"):
    print(f"  {r['case']:6s} frames {r['n_seg_frames']:5d}  centred {r['pct_centred']:3.0f}%  blur {r['pct_blur']:3.0f}%  "
          f"rel. speed {r['motion_rel']:.2f}  glare {r['pct_specular']:3.0f}%")

print("\nS9: C3VD (CAT5_external_validation.json)")
c = load("runs/own_data/renders/new/CAT5_external_validation.json")
med = [s["surface_median_mm"] for s in c["c3vd"]]
check("C3VD pooled median of sequence medians (mm)", float(np.median(med)), 1.40, 0.005)
check("C3VD pooled mean of sequence medians (mm)", float(np.mean(med)), 1.47, 0.005)

print(f"\n{'ALL CHECKS PASS' if not fails else f'{len(fails)} CHECK(S) FAILED: ' + '; '.join(fails)}")
sys.exit(1 if fails else 0)
