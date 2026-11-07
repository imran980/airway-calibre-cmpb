"""CT tracheal ground truth via TotalSegmentator (HU-threshold heuristics are not reliable across these
heterogeneous scans). Takes a thin axial series from the DICOM zip, converts it to NIfTI (HU, RAS affine),
runs TotalSegmentator and computes a tangent-corrected CSA / D_CE profile along the smoothed slice-centroid
centerline.

Gotchas: TMPDIR must be short (nnU-Net binds a Unix socket, 108-char limit); run the full total task
(--roi_subset trachea fails: "Crop is empty"); feed HU-normalised data; don't use a skeletonize centerline
(it collapses on a near-straight tube).

Usage: ct_gt_ts.py <CASE> <zip> [--series SERxxxxx] [--validate]"""
import sys, os, json, zipfile, shutil, subprocess
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])          # repository root
CASE = sys.argv[1]; ZIP = sys.argv[2]
SERIES = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--series"), None)
VALIDATE = "--validate" in sys.argv
SCR = os.environ.get("CT_SCRATCH", "ct_scratch")                    # working folder for unzipped DICOM / NIfTI
TSBIN = os.environ.get("TOTALSEG_BIN", "TotalSegmentator")
OUT = f"{ROOT}/runs/own_data/ct_gt"; os.makedirs(OUT, exist_ok=True)
import numpy as np, pydicom, nibabel as nib
from scipy import ndimage as ndi


def pick_series(zf):
    names = [n for n in zf.namelist() if "/SER" in n and not n.endswith("/")]
    sers = {}
    for n in names: sers.setdefault("SER" + n.split("/SER")[1].split("/")[0], []).append(n)
    best = None
    for s, files in sorted(sers.items()):
        if len(files) < 60: continue
        try:
            with zf.open(sorted(files)[len(files) // 2]) as fh: d = pydicom.dcmread(fh, stop_before_pixels=True, force=True)
            thk = float(getattr(d, "SliceThickness", 99) or 99)
            iop = [float(x) for x in (getattr(d, "ImageOrientationPatient", None) or [0] * 6)]
            if not (abs(abs(iop[0]) - 1) < 0.1 and abs(abs(iop[4]) - 1) < 0.1) or thk > 1.5: continue
            desc = str(getattr(d, "SeriesDescription", ""))
            if "EXP" in desc.upper() and "INSP" not in desc.upper(): continue      # prefer the inspiratory phase
            key = (thk, -len(files))
            if best is None or key < best[0]: best = (key, s, files, thk, desc)
        except Exception: continue
    if best is None: raise SystemExit(f"{CASE}: no thin axial series")
    print(f"{CASE}: series {best[1]} ({len(best[2])} slices, {best[3]} mm) '{best[4]}'", flush=True)
    return best[1], best[2]


def to_nifti(zf, files, work, nii):
    os.makedirs(work, exist_ok=True); sl = []
    for n in files:
        p = os.path.join(work, os.path.basename(n))
        with zf.open(n) as fh, open(p, "wb") as o: shutil.copyfileobj(fh, o)
        try:
            d = pydicom.dcmread(p, force=True)
            if not hasattr(d, "PixelData"): continue
            hu = d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
            sl.append((float(d.ImagePositionPatient[2]), hu, d))
        except Exception: continue
    if len(sl) < 30: raise SystemExit(f"{CASE}: only {len(sl)} readable slices")
    sl.sort(key=lambda s: s[0])
    d0 = sl[0][2]; py, px = [float(v) for v in d0.PixelSpacing]
    dz = float(np.median(np.diff([s[0] for s in sl])))
    vol = np.stack([s[1] for s in sl])                       # (z, y, x)
    if vol.min() > -500: vol = vol - 1024.0                  # some scans store un-shifted values
    ipp = [float(v) for v in d0.ImagePositionPatient]
    arr = np.transpose(vol, (2, 1, 0))                       # -> (x, y, z) for NIfTI
    aff = np.array([[-px, 0, 0, -ipp[0]], [0, -py, 0, -ipp[1]], [0, 0, dz, ipp[2]], [0, 0, 0, 1.0]])  # LPS -> RAS
    nib.save(nib.Nifti1Image(arr.astype(np.int16), aff), nii)
    print(f"  volume {vol.shape} spacing ({abs(dz):.2f},{py:.3f},{px:.3f}) mm  HU [{vol.min():.0f},{vol.max():.0f}] -> {os.path.basename(nii)}", flush=True)
    return (abs(dz), py, px)


def profile_from_mask(mask, spacing):
    """centerline = per-slice centroid of the lumen (smoothed); CSA = axial area x cos(tangent, z)."""
    dz, py, px = spacing; amm = py * px
    # per slice: largest component + whether the lumen is still single (below the carina it splits in two)
    per = []
    for zi in range(mask.shape[0]):
        m = mask[zi]
        if m.sum() < 4: per.append(None); continue
        lab, n = ndi.label(m)
        sizes = ndi.sum(m, lab, range(1, n + 1)) if n else []
        k = 1 + int(np.argmax(sizes)); big = lab == k
        single = (n == 1) or (sorted(sizes)[-2] <= 0.35 * max(sizes))
        ys, xs = np.where(big)
        per.append(dict(single=single, row=(zi * dz, ys.mean() * py, xs.mean() * px, big.sum() * amm)))
    # Follow the lumen from the top slice down to the carina, bridging gaps of up to GAP slices
    # (the longest contiguous run picks the wrong piece when a stenosis splits the lumen).
    ok = [bool(p and p["single"]) for p in per]
    masked = [bool(p) for p in per]
    GAP = 12
    top = next((k for k in range(len(per) - 1, -1, -1) if ok[k]), None)
    if top is None: raise SystemExit(f"{CASE}: no single-lumen slice")
    seq = [top]; k = top
    while k - 1 >= 0:
        nxt = next((q for q in range(k - 1, max(k - 1 - GAP, -1), -1) if ok[q]), None)
        if nxt is None: break                                   # gap too long, or the carina (persistent split)
        seq.extend(range(k - 1, nxt - 1, -1)); k = nxt
    seq = sorted(seq)
    i0, i1, n_run = seq[0], seq[-1], len(seq)
    rows, ngap = [], 0
    for k in seq:
        if per[k] and per[k]["single"]: rows.append(per[k]["row"])
        else:                                                   # closed / unsegmentable lumen: real zero-area station
            prev = rows[-1] if rows else (k * dz, 0.0, 0.0, 0.0)
            rows.append((k * dz, prev[1], prev[2], 0.0)); ngap += 1
    print(f"  lumen run (superior->carina): slices {i0}-{i1} ({n_run} spanned, {ngap} closed/unsegmented, {sum(masked)} masked)", flush=True)
    if len(rows) < 20: raise SystemExit(f"{CASE}: mask too short ({len(rows)} slices)")
    T = np.array(rows); C = T[:, :3].copy(); A = T[:, 3]
    for k in (1, 2): C[:, k] = ndi.uniform_filter1d(C[:, k], 9)      # smooth lateral wander, keep z exact
    d = np.gradient(C, axis=0); tang = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(ndi.uniform_filter1d(np.abs(tang[:, 0]), 9), 0.5, 1.0)
    csa = ndi.median_filter(A * cos, size=7)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    return arc, csa, 2 * np.sqrt(np.clip(csa, 0, None) / np.pi), C


zf = zipfile.ZipFile(ZIP)
ser, files = (SERIES, sorted(n for n in zf.namelist() if f"/{SERIES}/" in n and not n.endswith("/"))) if SERIES else pick_series(zf)
work = f"{SCR}/ctwork/{CASE}"; shutil.rmtree(work, ignore_errors=True)
nii = f"{work}/{CASE}.nii.gz"; os.makedirs(work, exist_ok=True)
spacing = to_nifti(zf, sorted(files), f"{work}/dcm", nii)
shutil.rmtree(f"{work}/dcm", ignore_errors=True)

env = dict(os.environ, TOTALSEG_HOME_DIR=f"{SCR}/ts/home", TMPDIR="/dev/shm/tst", TEMP="/dev/shm/tst", TMP="/dev/shm/tst")
os.makedirs("/dev/shm/tst", exist_ok=True)
seg = f"{work}/seg"
print("  running TotalSegmentator ...", flush=True)
r = subprocess.run([TSBIN, "-i", nii, "-o", seg], env=env, capture_output=True, text=True)   # full task; --roi_subset under-segments
if not os.path.exists(f"{seg}/trachea.nii.gz"):
    print(r.stdout[-1500:]); print(r.stderr[-2500:]); raise SystemExit(f"{CASE}: TotalSegmentator produced no trachea mask")

m = nib.load(f"{seg}/trachea.nii.gz").get_fdata() > 0.5            # (x, y, z)
mask = np.transpose(m, (2, 1, 0))                                   # -> (z, y, x)
print(f"  trachea mask: {mask.sum()} voxels = {mask.sum()*spacing[0]*spacing[1]*spacing[2]/1000:.1f} cm^3", flush=True)
arc, csa, dce, C = profile_from_mask(mask, spacing)
print(f"  profile: arclength {arc.max():.0f} mm over {len(arc)} stations, D_CE median {np.median(dce):.1f} mm (range {dce.min():.1f}-{dce.max():.1f})", flush=True)

if VALIDATE:
    z = np.load(f"{ROOT}/runs/own_data/ct_validated_3/ts_{CASE}/gt_{CASE}.json/ground_truth.npz")
    ra, rd = np.asarray(z["arclength_mm"]), np.asarray(z["dce_mm"]); best = None
    # the two extractions cover different sub-spans of the same trachea, so align by the best shift (and flip)
    L = min(arc.max(), ra.max())
    for flip in (False, True):
        a2, d2 = (arc.max() - arc[::-1], dce[::-1]) if flip else (arc, dce)
        for sh in np.linspace(0, max(ra.max() - L, 0), 41):
            u = np.linspace(0, L, 120)
            g, r2 = np.interp(u, a2, d2), np.interp(u + sh, ra, rd)
            rm = float(np.sqrt(np.mean((g - r2) ** 2)))
            if best is None or rm < best[0]: best = (rm, float(np.mean(g - r2)), float(np.corrcoef(g, r2)[0, 1]), flip, L, float(sh))
    rm, bi, cc, flip, L, sh = best
    print(f"  VALIDATE vs saved GT: arclength {arc.max():.0f} vs {ra.max():.0f} mm | median D_CE {np.median(dce):.2f} vs {np.median(rd):.2f} mm")
    print(f"    best-aligned over {L:.0f} mm (flip={flip}, shift {sh:.0f} mm): RMSE {rm:.2f} mm, bias {bi:+.2f} mm, r {cc:.2f}")
else:
    np.savez(f"{OUT}/gt_{CASE}.npz", arclength_mm=arc, csa_mm2=csa, dce_mm=dce, centerline_mm=C)
    json.dump(dict(case=CASE, series=ser, zip=os.path.basename(ZIP), n_stations=int(len(arc)), arclength_mm=float(arc.max()),
                   dce_median_mm=float(np.median(dce)), dce_min_mm=float(dce.min()), dce_max_mm=float(dce.max()),
                   method="TotalSegmentator trachea + smoothed-centroid centerline + tangent-corrected CSA"),
              open(f"{OUT}/gt_{CASE}.json", "w"), indent=2)
    print(f"  saved ct_gt/gt_{CASE}.npz + .json")
shutil.rmtree(work, ignore_errors=True)
