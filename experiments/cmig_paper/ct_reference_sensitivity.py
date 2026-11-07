"""Sensitivity of the gated calibre RMSE to the CT reference (20-V1).

Registers the 20-V1 gated profile (from ct3_regate.json, stored scale divided out) to the TotalSegmentator
reference and to a fixed -900 HU threshold reference with the same constrained isotropic Sim(3). The two
references differ by ~1.4 mm in D_CE (median CSA 48.9 vs 33.5 mm2); 50-V2 already uses the threshold.
Re-registering against TotalSegmentator gives 0.92 mm (Table 3: 0.83 mm), so compare the rows with each
other rather than with Table 3.

    python ct_reference_sensitivity.py --raw <ct_raw_insp.npz> --case 20-V1
"""
import argparse, json
import numpy as np


def robust_register(a_r, r_m, ct_arc, ct_dce, min_frac=0.6):
    """Constrained isotropic Sim(3): one scale + orientation, profile shape not fitted."""
    a_r = a_r - a_r[0]
    o = np.argsort(a_r); a_r, r_m = a_r[o], r_m[o]
    rmed = float(np.median(r_m))
    s_lo, s_hi = 0.5 * ct_dce.min() / (2 * rmed), 2.0 * ct_dce.max() / (2 * rmed)
    best = None
    for flip in (False, True):
        C = ct_dce[::-1] if flip else ct_dce
        A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
        for s in np.linspace(s_lo, s_hi, 1500):
            e = 2 * s * r_m - np.interp(s * a_r, A, C, left=np.nan, right=np.nan)
            m = np.isfinite(e)
            if m.sum() < max(6, min_frac * len(a_r)):
                continue
            rm = float(np.sqrt(np.mean(e[m] ** 2)))
            if best is None or rm < best[0]:
                best = (rm, s, flip)
    if best is None:
        return None
    _, s, flip = best
    C = ct_dce[::-1] if flip else ct_dce
    A = (ct_arc[-1] - ct_arc[::-1]) if flip else ct_arc
    dce, am = 2 * s * r_m, s * a_r
    ct = np.interp(am, A, C, left=np.nan, right=np.nan)
    m = np.isfinite(ct); err = dce[m] - ct[m]
    return dict(scale=float(s), rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(np.mean(err)),
                n=int(m.sum()), span_mm=float(am[m].ptp()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="npz with z, area, dce from the -900 HU measurement")
    ap.add_argument("--case", default="20-V1")
    ap.add_argument("--gt", default="runs/own_data/ct_validated_3/ts_20-V1/gt_20-V1.json/ground_truth.npz")
    ap.add_argument("--regate", default="runs/own_data/reports/ct3_regate.json")
    a = ap.parse_args()

    g = json.load(open(a.regate))[a.case]["circle-only"]
    s0 = g["scale"]
    a_r, r_m = np.array(g["am"]) / s0, np.array(g["dce"]) / (2 * s0)

    rows = {}
    gt = np.load(a.gt)
    R = robust_register(a_r, r_m, np.asarray(gt["arclength_mm"], float), np.asarray(gt["dce_mm"], float))
    rows["TotalSegmentator (published reference)"] = R

    d = np.load(a.raw); z, dce = d["z"], d["dce"]
    brk = np.nonzero(np.abs(np.diff(z)) > 2)[0]
    for b in np.split(np.arange(len(z)), brk + 1):
        if len(b) < 8:
            continue
        arc = np.r_[0, np.cumsum(np.abs(np.diff(z[b])))]
        rows[f"raw -900 HU, z {z[b].max():.0f}-{z[b].min():.0f} ({len(b)} levels)"] = \
            robust_register(a_r, r_m, arc, dce[b])

    print(f"{'reference':44s} {'RMSE':>6s} {'bias':>7s} {'n':>4s} {'scale':>7s} {'span':>8s}")
    for k, v in rows.items():
        if v is None:
            print(f"{k:44s}   no valid fit"); continue
        print(f"{k:44s} {v['rmse']:6.2f} {v['bias']:+7.2f} {v['n']:4d} {v['scale']:7.3f} {v['span_mm']:7.1f}mm")
    p = f"runs/own_data/reports/ct_reference_sensitivity_{a.case}.json"
    json.dump(rows, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
