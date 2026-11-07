"""Per-session calibration for defocused / low-contrast board videos (e.g. 17-V1).
Forces the cohort board (14x13 inner corners) instead of searching patterns (searching lets a spurious 7x5 sub-grid win on a
soft board -> garbage intrinsics). Detection is tried per frame at full resolution, then on a 0.5x downscaled + unsharp-masked
copy (corners scaled back), then with the classic detector + cornerSubPix. Sanity gate before writing: fy/fx within 2% of 1,
HFOV 60-115 deg, RMS reported (strict pass < 0.5 px, marginal above). Same output format as calibrate_case.py.
Usage: python calibrate_case_robust.py CASE [CASE ...]   [--force to overwrite an existing PASS file]"""
import sys, os, glob, json, cv2, numpy as np
DATA_ROOT = os.environ.get("BRONCHO_DATA", "data")   # calibration videos live under here

BOARD = (14, 13); STEP = 6; MAXF = 400
force = "--force" in sys.argv; cases = [c for c in sys.argv[1:] if not c.startswith("--")]
SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
CL = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def detect(gray):
    ok, cn = cv2.findChessboardCornersSB(gray, BOARD, SB)
    if ok: return cn, "sb"
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sharp = cv2.addWeighted(small, 1.8, cv2.GaussianBlur(small, (0, 0), 2.0), -0.8, 0)
    ok, cn = cv2.findChessboardCornersSB(sharp, BOARD, SB)
    if ok: return cn * 2.0, "sb-half-unsharp"
    ok, cn = cv2.findChessboardCorners(gray, BOARD, CL)
    if ok:
        cn = cv2.cornerSubPix(gray, cn, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)); return cn, "classic"
    return None, None


for c in cases:
    out = f"runs/own_data/retry/{c}_intrinsics.json"
    try:
        prev = json.load(open(out))
        if not force and prev.get("rms_reproj_px", 9) < 0.5 and abs(prev.get("fy_over_fx", 0) - 1) < 0.02: print(f"{c}: existing PASS kept ({prev['rms_reproj_px']:.3f} px); use --force to redo"); continue
    except FileNotFoundError: pass
    vids = glob.glob(f"{DATA_ROOT}/**/{c}*alib*.[mM][pP]4", recursive=True)
    if not vids: print(f"{c}: NO CALIB VIDEO"); continue
    vid = vids[0]; cap = cv2.VideoCapture(vid); obj, img, how, fi, w, h = [], [], {}, 0, None, None
    objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32); objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2)
    while len(obj) < MAXF:
        ok, f = cap.read()
        if not ok: break
        if fi % STEP == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); h, w = g.shape; cn, mode = detect(g)
            if cn is not None: obj.append(objp); img.append(cn.astype(np.float32)); how[mode] = how.get(mode, 0) + 1
        fi += 1
    cap.release()
    if len(obj) < 10: print(f"{c}: only {len(obj)} board detections of {BOARD} -> FAIL ({vid.split('/')[-1]})"); continue
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (w, h), None, None, flags=cv2.CALIB_FIX_K3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]; k1, k2, p1, p2 = dist.ravel()[:4]
    hf = 2 * np.degrees(np.arctan(w / (2 * fx))); vf = 2 * np.degrees(np.arctan(h / (2 * fy))); r = fy / fx
    sane = abs(r - 1) < 0.02 and 60 < hf < 115      # cohort optics: ~97-99 deg, ~106 deg, ~78 deg (24-V2); isotropy is the real garbage detector
    tag = "FAIL (insane geometry, not written)" if not sane else ("PASS" if rms < 0.5 else "MARGINAL")
    print(f"{c}: board {BOARD} frames {len(obj)} {how} RMS {rms:.3f}px fx {fx:.1f} fy/fx {r:.4f} HFOV {hf:.1f} -> {tag}", flush=True)
    if not sane: continue
    d = dict(model="OPENCV", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2, params_colmap=[fx, fy, cx, cy, k1, k2, p1, p2],
             rms_reproj_px=rms, hfov_deg=hf, vfov_deg=vf, fy_over_fx=r, n_frames=len(obj), pattern=list(BOARD), detection_modes=how,
             source_calib_video=vid, refine=False, method="calibrate_case_robust (forced 14x13; SB full-res -> 0.5x unsharp -> classic)")
    json.dump(d, open(out, "w"), indent=2)
