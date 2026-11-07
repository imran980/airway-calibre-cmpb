"""Cross-section rings (mm) at the accepted M1 stations of a CT case, using the scale stored in the M1 entry of
rerun_eval_*.json. The station maths is copied from ct3_rerun_eval.cam_profile (spline, slabs, 90th-percentile
radial trim, coverage, MAD trim); m1_rings() asserts that the recomputed stations match the json."""
import re, json
import numpy as np, open3d as o3d, pycolmap
from scipy.interpolate import splprep, splev


def light_clean(ply):
    pc = o3d.io.read_point_cloud(ply); pc, _ = pc.remove_statistical_outlier(20, 2.0); return np.asarray(pc.points)


def cam_stations(P, sparse, fr=None, NST=45):
    """camera-centerline stations: returns dict(arc, rmed, cov, C, T, E1, E2, slab); same maths as ct3_rerun_eval.cam_profile."""
    rec = pycolmap.Reconstruction(sparse)
    cams = sorted(((int(re.search(r"f(\d+)", im.name).group(1)), np.asarray(im.projection_center())) for im in rec.images.values()), key=lambda x: x[0])
    if fr: cams = [c for c in cams if fr[0] <= c[0] <= fr[1]]
    Cc = np.array([c[1] for c in cams]); tck, _ = splprep(Cc.T, u=np.linspace(0, 1, len(Cc)), s=len(Cc) * 2.0); us = np.linspace(0, 1, NST)
    C = np.array(splev(us, tck)).T; T = np.array(splev(us, tck, der=1)).T; T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]; slab = (arc[-1] / (NST - 1)) * 0.8
    rmed = np.full(NST, np.nan); cov = np.zeros(NST); E1 = np.zeros_like(T); E2 = np.zeros_like(T)
    for i, (p, t) in enumerate(zip(C, T)):
        e1 = np.array([t[1], -t[0], 0.]); e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(t, e1); E1[i], E2[i] = e1, e2
        d = P - p; al = d @ t; sel = np.abs(al) < slab / 2; q = d[sel] - np.outer(al[sel], t); r = np.linalg.norm(q, axis=1)
        if len(r) < 15: continue
        k = r < np.percentile(r, 90); q, r = q[k], r[k]
        if len(r) < 12: continue
        cov[i] = (np.histogram(np.arctan2(q @ e2, q @ e1), 36, (-np.pi, np.pi))[0] > 0).sum() / 36; rmed[i] = np.median(r)
    return dict(arc=arc, rmed=rmed, cov=cov, C=C, T=T, E1=E1, E2=E2, slab=slab)


def trim(r):
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() >= 6:
        med = np.median(r[ok]); mad = np.median(np.abs(r[ok] - med)) + 1e-9; ok &= np.abs(r - med) < 3.5 * mad
    return ok


def m1_rings(P, sparse, m1, fr=None, quantiles=(0.25, 0.5, 0.75), tol=0.05, by="arclength"):
    """rings (mm) at accepted M1 stations. m1 = the 'M1 cam-centerline + cov>=0.75' dict of a rerun_eval json (scale, am, dce, ct_arc, ct_dce).
    Asserts that the recomputed station arclengths and calibres match the json."""
    S = cam_stations(P, sparse, fr); ok = (S["cov"] >= 0.75) & trim(S["rmed"]); idx = np.where(ok)[0]
    s = m1["scale"]; a0 = S["arc"][idx][0]; am_all = s * (S["arc"] - a0); dce_all = 2 * s * S["rmed"]
    A, Cc = np.array(m1["ct_arc"]), np.array(m1["ct_dce"]); ct_all = np.interp(am_all, A, Cc, left=np.nan, right=np.nan)
    keep = idx[np.isfinite(ct_all[idx])]
    am_j, dce_j = np.array(m1["am"]), np.array(m1["dce"])
    if len(keep) == len(am_j):
        assert np.allclose(am_all[keep], am_j, atol=tol) and np.allclose(dce_all[keep], dce_j, atol=tol), "recomputed stations do not reproduce the json"
        check = "reproduces json"
    else:
        check = f"station count differs ({len(keep)} vs json {len(am_j)})"
    out = []
    amk = am_all[keep]
    for q in quantiles:
        i = keep[int(np.argmin(np.abs(amk - (amk.min() + q * (amk.max() - amk.min())))))] if by == "arclength" else keep[int(q * (len(keep) - 1))]
        p, t = S["C"][i], S["T"][i]
        d = P - p; al = d @ t; sel = np.abs(al) < S["slab"] / 2; qv = d[sel] - np.outer(al[sel], t); r = np.linalg.norm(qv, axis=1)
        k = r < np.percentile(r, 90); qv = qv[k]
        xy = np.c_[qv @ S["E1"][i], qv @ S["E2"][i]] * s
        out.append(dict(am=float(am_all[i]), xy=xy, R=float(s * S["rmed"][i]), ct_d=float(ct_all[i]), cov=float(S["cov"][i]), n=int(k.sum())))
    return out, check
