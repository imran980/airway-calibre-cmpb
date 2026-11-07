"""20-V1 inspiratory/expiratory raw-CT lumen area at matched anatomical levels.

The TotalSegmentator trachea mask only seeds the lumen; the area is the -900 HU air component
grown from that seed on the raw axial slice. Slices are paired between phases by
ImagePositionPatient z (NIfTI RAS -> DICOM LPS sign flip).

  python ct_phase_pair_raw.py --insp <SER00005 dir> --exp <SER00011 dir> --mask <trachea.nii.gz>

Prints per-level areas and writes runs/own_data/reports/ct_phase_pair_20-V1.json.
"""
import argparse, json, os
import nibabel as nib, numpy as np, pydicom
from scipy import ndimage as ndi

AIR_HU = -900.0
MAX_MM2 = 600.0    # larger than this means the grow spilled into lung
MIN_MM2 = 8.0      # smaller means the seed failed or the run hit the carina


def load_series(d):
    sl = [ds for ds in (pydicom.dcmread(os.path.join(d, f)) for f in sorted(os.listdir(d)))
          if hasattr(ds, "PixelData")]
    sl.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    V = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1))
                  + float(getattr(s, "RescaleIntercept", 0)) for s in sl])
    z = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    org = np.array([[float(v) for v in s.ImagePositionPatient] for s in sl])
    return V, z, org, float(sl[0].PixelSpacing[0])


def grow(img, cy, cx, px):
    """Area of the air component containing (cy,cx), nudging onto air within 8 px."""
    air = img < AIR_HU
    if not (0 <= cy < img.shape[0] and 0 <= cx < img.shape[1]):
        return None
    if not air[cy, cx]:
        w = air[max(cy - 8, 0):cy + 9, max(cx - 8, 0):cx + 9]
        if not w.any():
            return None
        dy, dx = np.nonzero(w)
        j = int(np.argmin((dy - 8) ** 2 + (dx - 8) ** 2))
        cy, cx = max(cy - 8, 0) + dy[j], max(cx - 8, 0) + dx[j]
    lab, _ = ndi.label(air)
    a = float((lab == lab[cy, cx]).sum()) * px
    return None if a > MAX_MM2 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insp", required=True); ap.add_argument("--exp", required=True)
    ap.add_argument("--mask", required=True); ap.add_argument("--case", default="20-V1")
    a = ap.parse_args()

    seg = nib.load(a.mask); M = np.asanyarray(seg.dataobj) > 0
    Vi, zi, orgi, psi = load_series(a.insp)
    Ve, ze, orge, pse = load_series(a.exp)
    pxi, pxe = psi * psi, pse * pse

    rows, seeds = [], 0
    for k in np.nonzero(M.any(axis=(0, 1)))[0]:
        ii, jj = np.nonzero(M[:, :, k])
        P = nib.affines.apply_affine(seg.affine,
                                     np.stack([ii, jj, np.full(len(ii), k)], 1).astype(float))
        xl, yl, zc = -P[:, 0].mean(), -P[:, 1].mean(), P[:, 2].mean()   # RAS -> LPS
        ki, ke = int(np.argmin(np.abs(zi - zc))), int(np.argmin(np.abs(ze - zc)))
        ai = grow(Vi[ki], int(round((yl - orgi[ki][1]) / psi)), int(round((xl - orgi[ki][0]) / psi)), pxi)
        ae = grow(Ve[ke], int(round((yl - orge[ke][1]) / pse)), int(round((xl - orge[ke][0]) / pse)), pxe)
        if ai is not None:
            seeds += 1
        if ai is not None and ae is not None:
            rows.append(dict(z_mm=float(zc), insp_mm2=ai, exp_mm2=ae))

    good = [r for r in rows if r["insp_mm2"] >= MIN_MM2 and r["exp_mm2"] >= MIN_MM2]
    I = np.array([r["insp_mm2"] for r in good]); E = np.array([r["exp_mm2"] for r in good])
    Z = np.array([r["z_mm"] for r in good]); per = 1 - E / I
    j = int(np.argmax(I))
    out = dict(case=a.case, threshold_hu=AIR_HU, n_matched=len(good),
               z_range_mm=[float(Z.min()), float(Z.max())], span_mm=float(Z.ptp()),
               insp_median_mm2=float(np.median(I)), exp_median_mm2=float(np.median(E)),
               reduction_of_medians_pct=float(100 * (1 - np.median(E) / np.median(I))),
               per_level_mean_pct=float(100 * per.mean()), per_level_max_pct=float(100 * per.max()),
               widest_level=dict(z_mm=float(Z[j]), insp_mm2=float(I[j]), exp_mm2=float(E[j]),
                                 reduction_pct=float(100 * (1 - E[j] / I[j]))),
               levels=good)
    print(f"seeds landing on air (insp): {seeds}; matched levels kept {len(good)}/{len(rows)}")
    print(f"z {Z.min():.1f}-{Z.max():.1f} mm | INSP {np.median(I):.1f} -> EXP {np.median(E):.1f} mm^2 "
          f"= {out['reduction_of_medians_pct']:.1f}%")
    print(f"widest level z={Z[j]:.1f}: {I[j]:.1f} -> {E[j]:.1f} mm^2 = {out['widest_level']['reduction_pct']:.1f}%")
    p = f"runs/own_data/reports/ct_phase_pair_{a.case}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
