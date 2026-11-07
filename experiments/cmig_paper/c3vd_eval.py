"""C3VD validation. Sim3-align the SfM camera trajectory to the GT poses (pose.txt), transform the
dense cloud into GT world mm and score it against coverage_mesh.obj:
  - pose residual after the Sim3 alignment (mm)
  - accuracy: cloud->mesh distance (mm)
  - completeness: mesh->cloud distance (mm)
Also writes an overlay plot (GT mesh grey, reconstructed cloud colored by error).
Usage: python c3vd_eval.py <seq_dir> <sparse_dir> <fused.ply> <label> <out.png>
"""
import sys, re, numpy as np, open3d as o3d, pycolmap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

seq, sparse, fused, label, outpng = sys.argv[1:6]

# GT camera centers (pose.txt: 4x4 row-major [[R,0],[t,1]] in mm; center = translation row)
poses = np.loadtxt(f"{seq}/pose.txt", delimiter=",")
gt_center = poses[:, 12:15]

# SfM camera centers keyed by original frame index (name = "N_color.png")
rec = pycolmap.Reconstruction(sparse)
sfm = {}
for im in rec.images.values():
    mt = re.search(r"(\d+)", im.name.split("/")[-1])       # frame idx: fNNNNN.png or N_color.png
    if mt: sfm[int(mt.group(1))] = np.asarray(im.projection_center())
idx = sorted(k for k in sfm if k < len(gt_center))
A = np.array([sfm[i] for i in idx]); B = gt_center[idx]


def umeyama(A, B):
    mA, mB = A.mean(0), B.mean(0); AA, BB = A - mA, B - mB
    H = AA.T @ BB / len(A); U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T; s = (S * [1, 1, d]).sum() / ((AA ** 2).sum() / len(A))
    return s, R, mB - s * R @ mA


s, R, t = umeyama(A, B)
resid = np.linalg.norm((s * (A @ R.T) + t) - B, axis=1)
print(f"[{label}] Sim3 on {len(idx)} cams: scale={s:.4g} | pose resid median={np.median(resid):.2f}mm mean={resid.mean():.2f}mm")

# reconstructed cloud -> GT world mm
P = np.asarray(o3d.io.read_point_cloud(fused).points)
m = np.median(P, 0); dd = np.linalg.norm(P - m, axis=1)
P = P[dd < np.median(dd) + 4 * np.median(np.abs(dd - np.median(dd)))]
Pw = (s * (P @ R.T) + t).astype(np.float32)

# GT mesh and distances
mesh = o3d.io.read_triangle_mesh(f"{seq}/coverage_mesh.obj"); mesh.compute_vertex_normals()
scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
acc = scene.compute_distance(o3d.core.Tensor(Pw)).numpy()                     # cloud->mesh (accuracy)
from scipy.spatial import cKDTree
mp = np.asarray(mesh.sample_points_uniformly(60000).points)
comp, _ = cKDTree(Pw).query(mp, k=1)                                          # mesh->cloud (completeness)
d_cam, _ = cKDTree(B).query(mp, k=1)                                          # mesh point -> nearest camera (world mm)
obs = d_cam < 35.0                                                            # only the mesh region the scope could see
co = comp[obs]
print(f"  ACCURACY  cloud->mesh: median={np.median(acc):.2f}mm mean={acc.mean():.2f} p90={np.percentile(acc,90):.2f} | {len(Pw)} pts")
print(f"  COMPLETE  observed ({obs.mean():.0%} of mesh within 35mm of path): median={np.median(co):.2f}mm <1mm={np.mean(co<1):.0%} <2mm={np.mean(co<2):.0%} | all-mesh median={np.median(comp):.1f}mm")

# overlay: GT mesh (grey) + cloud colored by accuracy error
V = np.asarray(mesh.vertices); Tri = np.asarray(mesh.triangles)
ctr = V.mean(0); _, _, Vt = np.linalg.svd(V - ctr, full_matrices=False)
def proj(X): Q = (X - ctr) @ Vt.T; return Q[:, 0], Q[:, 1]
sub = np.random.default_rng(0).choice(len(Pw), min(60000, len(Pw)), replace=False)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
mx, my = proj(V); ax[0].triplot(mx, my, Tri[::20], color="0.8", lw=0.2)
px, py = proj(Pw[sub]); sc = ax[0].scatter(px, py, s=1, c=np.clip(acc[sub], 0, 3), cmap="turbo")
ax[0].set_title(f"{label}: recon cloud (color=err mm) over GT mesh"); ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
fig.colorbar(sc, ax=ax[0], label="cloud->mesh err (mm)")
ax[1].hist(acc, bins=80, range=(0, 5), color="#2b6cb0", alpha=0.8); ax[1].axvline(np.median(acc), color="r", ls="--", label=f"median {np.median(acc):.2f}mm")
ax[1].set_title("accuracy distribution"); ax[1].set_xlabel("cloud->mesh distance (mm)"); ax[1].legend()
fig.tight_layout(); fig.savefig(outpng, dpi=120); print(f"  saved {outpng}")
