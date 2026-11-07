"""Tracheal lumen area per axial CT slice from a fixed HU threshold (no segmentation model).

One row per slice; the carina is also detected so the two respiratory phases can be compared at
the same level. Used for the 20-V1 inspiratory/expiratory areas (Table S3).

    python ct_raw_lumen.py <series_dir> [--label INSP] [--thr -900]

Method (same settings for every case):
  body mask   : HU > -500, largest connected component, holes filled  (excludes room air)
  lumen       : HU < threshold (default -900) inside the body mask
  trachea     : seeded on the most superior slice as the in-body air component nearest the midline
                with area 8-400 mm^2 and circularity >= 0.5; then tracked inferiorly by centroid
                proximity, rejecting jumps > 8 mm or area changes > 4x
  carina      : first slice at which two in-body air components each >= 25% of the running tracheal
                area lie within 40 mm of the tracked centroid
  area        : pixel count x pixel area; D_CE = 2*sqrt(A/pi)
"""
import argparse
import json
import os
import sys

import numpy as np
import pydicom
from scipy import ndimage as ndi


def load_series(d):
    """Return HU volume (z, y, x) ordered superior->inferior, plus geometry."""
    sl = []
    for f in sorted(os.listdir(d)):
        try:
            ds = pydicom.dcmread(os.path.join(d, f))
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        sl.append(ds)
    if not sl:
        sys.exit(f"no readable slices in {d}")
    sl.sort(key=lambda s: float(s.ImagePositionPatient[2]), reverse=True)   # superior first
    px = float(sl[0].PixelSpacing[0]) * float(sl[0].PixelSpacing[1])
    z = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    vol = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1))
                    + float(getattr(s, "RescaleIntercept", 0)) for s in sl])
    return vol, z, px, sl[0]


def in_body_air(img, thr):
    """Air components strictly inside the patient (room air removed via a filled body mask)."""
    body = ndi.binary_fill_holes(img > -500)
    lab, n = ndi.label(body)
    if n > 1:
        body = lab == (1 + int(np.argmax(ndi.sum(body, lab, range(1, n + 1)))))
    body = ndi.binary_erosion(body, iterations=2)
    return (img < thr) & body


def components(mask, px, min_mm2=4.0):
    lab, n = ndi.label(mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        a = float(m.sum()) * px
        if a < min_mm2:
            continue
        ys, xs = np.nonzero(m)
        per = float(np.sum(m ^ ndi.binary_erosion(m))) or 1.0
        out.append(dict(area=a, cy=float(ys.mean()), cx=float(xs.mean()),
                        circ=min(1.0, 4 * np.pi * m.sum() / (per ** 2))))
    return out


def track(vol, z, px, thr):
    H, W = vol.shape[1:]
    mmx = np.sqrt(px)                     # ~mm per pixel
    rows, prev, running = [], None, None
    carina_i = None
    for i in range(vol.shape[0]):
        cs = components(in_body_air(vol[i], thr), px)
        if not cs:
            rows.append(None); continue
        if prev is None:
            cand = [c for c in cs if 8 <= c["area"] <= 400 and c["circ"] >= 0.5]
            if not cand:
                rows.append(None); continue
            c = min(cand, key=lambda c: abs(c["cx"] - W / 2))
        else:
            near = [c for c in cs if np.hypot(c["cx"] - prev["cx"], c["cy"] - prev["cy"]) * mmx < 8
                    and 0.25 * running < c["area"] < 4 * running]
            if not near:
                rows.append(None); continue
            c = min(near, key=lambda c: np.hypot(c["cx"] - prev["cx"], c["cy"] - prev["cy"]))
            if carina_i is None:
                big = [q for q in cs
                       if q["area"] >= 0.25 * running
                       and np.hypot(q["cx"] - prev["cx"], q["cy"] - prev["cy"]) * mmx < 40]
                if len(big) >= 2:
                    carina_i = i
        prev = c
        running = c["area"] if running is None else 0.7 * running + 0.3 * c["area"]
        rows.append(dict(i=i, z=float(z[i]), area=c["area"],
                         dce=2 * np.sqrt(c["area"] / np.pi), cx=c["cx"], cy=c["cy"]))
    return rows, carina_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--label", default="")
    ap.add_argument("--thr", type=float, default=-900)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    vol, z, px, ds = load_series(a.series)
    rows, carina = track(vol, z, px, a.thr)
    got = [r for r in rows if r]
    lab = a.label or getattr(ds, "SeriesDescription", a.series)
    print(f"\n=== {lab} ===  {vol.shape[0]} slices, pixel area {px:.5f} mm^2, "
          f"z {z.max():.1f} -> {z.min():.1f} mm, threshold {a.thr:g} HU")
    if carina is not None:
        print(f"  carina at slice index {carina} (z = {z[carina]:.1f} mm)")
    else:
        print("  carina not detected in this stack")
    print(f"  trachea tracked on {len(got)}/{vol.shape[0]} slices, "
          f"z {max(r['z'] for r in got):.1f} -> {min(r['z'] for r in got):.1f} mm")
    out = dict(series=a.series, label=lab, threshold_hu=a.thr, pixel_area_mm2=px,
               carina_index=carina, carina_z=(float(z[carina]) if carina is not None else None),
               slices=got)
    p = a.out or f"runs/own_data/reports/rawlumen_{lab.replace(' ', '_')}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
