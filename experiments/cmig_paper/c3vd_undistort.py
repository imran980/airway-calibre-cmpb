"""Undistort C3VD 195-deg fisheye frames to a central pinhole (default 120-deg FOV) so COLMAP MVS works.
Builds a theta->rho LUT from the Scaramuzza model, remaps every *_color.png, writes pinhole images +
the pinhole OPENCV intrinsics. Usage: python c3vd_undistort.py <seq_dir> <out_img_dir> [fov_deg=120] [size=1024]"""
import sys, os, re, glob, json, numpy as np, cv2

seq, out = sys.argv[1], sys.argv[2]
FOV = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
S = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
os.makedirs(out, exist_ok=True)
for p in glob.glob(f"{out}/*.png"): os.unlink(p)

I = dict(cx=678.544839263292, cy=542.975887548343, a0=769.243600037458,
         a2=-0.000812770624150226, a3=6.25674244578925e-07, a4=-1.19662182144280e-09,
         c=0.999986882249990, d=0.00288273829525059, e=-0.00296316513429569)
St = np.array([[I["c"], I["d"]], [I["e"], 1.0]])                 # forward stretch uvpp->uvp

# theta(rho) LUT from Scaramuzza (monotonic region)
rho = np.linspace(0, 950, 4000)
z = I["a0"] + I["a2"] * rho ** 2 + I["a3"] * rho ** 3 + I["a4"] * rho ** 4
theta = np.arctan2(rho, z)
good = np.concatenate([[True], np.diff(theta) > 0])              # keep monotonic increasing part
rho, theta = rho[good], theta[good]

# target pinhole
f = (S / 2) / np.tan(np.radians(FOV / 2)); cx = cy = S / 2
u, v = np.meshgrid(np.arange(S), np.arange(S))
x = (u - cx) / f; y = (v - cy) / f; rr = np.hypot(x, y)
th = np.arctan(rr); phi = np.arctan2(y, x)
rr_fish = np.interp(th, theta, rho, right=np.nan)               # rho(theta); NaN beyond FOV
uvpp = np.stack([rr_fish * np.cos(phi), rr_fish * np.sin(phi)], -1)
uvp = uvpp @ St.T
map_x = (uvp[..., 0] + I["cx"]).astype(np.float32)
map_y = (uvp[..., 1] + I["cy"]).astype(np.float32)
valid = np.isfinite(rr_fish)

n = 0
for fp in glob.glob(f"{seq}/*_color.png"):
    idx = int(re.match(r"(\d+)_color", os.path.basename(fp)).group(1))
    img = cv2.imread(fp)
    warp = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    warp[~valid] = 0
    cv2.imwrite(f"{out}/f{idx:05d}.png", warp, [cv2.IMWRITE_PNG_COMPRESSION, 1]); n += 1
params = [float(f), float(f), float(cx), float(cy), 0.0, 0.0, 0.0, 0.0]         # pinhole OPENCV (no distortion)
json.dump({"camera_model": "OPENCV", "width": S, "height": S, "params_colmap": params,
           "fov_deg": FOV, "src": seq}, open(f"{out}/../pinhole_intrinsics.json", "w"), indent=2)
print(f"undistorted {n} frames -> {out} | pinhole {S}x{S} f={f:.1f} fov={FOV} | valid={valid.mean():.0%}")
