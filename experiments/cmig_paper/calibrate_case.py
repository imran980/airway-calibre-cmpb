"""Per-session calibration -> runs/own_data/retry/<case>_intrinsics.json (OPENCV, k3 fixed 0), format used by recover_clip.find_calib."""
import sys, os, glob, json, cv2, numpy as np
DATA_ROOT=os.environ.get("BRONCHO_DATA","data")  # calibration videos
for c in sys.argv[1:]:
    vids=glob.glob(f"{DATA_ROOT}/**/{c}*alib*.[mM][pP]4", recursive=True)
    if not vids: print(f"{c}: NO CALIB VIDEO"); continue
    vid=vids[0]; cap=cv2.VideoCapture(vid); frames=[]; fi=0
    while True:
        ok,f=cap.read()
        if not ok: break
        if fi%6==0: frames.append(f)
        fi+=1
    cap.release(); h,w=frames[0].shape[:2]; flags=cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY
    best=None
    for pat in [(14,13),(13,14),(9,6),(6,9),(7,5),(5,7),(10,7),(7,10),(8,6),(6,8)]:
        hits=sum(cv2.findChessboardCornersSB(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),pat,flags)[0] for f in frames[:24])
        if best is None or hits>best[1]: best=(pat,hits)
    pat=best[0]; objp=np.zeros((pat[0]*pat[1],3),np.float32); objp[:,:2]=np.mgrid[0:pat[0],0:pat[1]].T.reshape(-1,2); obj,img=[],[]
    for f in frames:
        ok2,cn=cv2.findChessboardCornersSB(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),pat,flags)
        if ok2: obj.append(objp); img.append(cn)
    if len(obj)<10: print(f"{c}: only {len(obj)} board detections (pattern {pat}) -> FAIL"); continue
    rms,K,dist,_,_=cv2.calibrateCamera(obj,img,(w,h),None,None,flags=cv2.CALIB_FIX_K3)
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]; k1,k2,p1,p2=dist.ravel()[:4]; hf=2*np.degrees(np.arctan(w/(2*fx)))
    d=dict(model="OPENCV",width=w,height=h,fx=fx,fy=fy,cx=cx,cy=cy,k1=k1,k2=k2,p1=p1,p2=p2,params_colmap=[fx,fy,cx,cy,k1,k2,p1,p2],rms_reproj_px=rms,hfov_deg=hf,vfov_deg=2*np.degrees(np.arctan(h/(2*fy))),fy_over_fx=fy/fx,n_frames=len(obj),pattern=list(pat),source_calib_video=vid,refine=False)
    json.dump(d,open(f"runs/own_data/retry/{c}_intrinsics.json","w"),indent=2)
    print(f"{c}: pattern {pat} frames {len(obj)} RMS {rms:.3f}px fx {fx:.1f} fy/fx {fy/fx:.4f} HFOV {hf:.1f} -> {'PASS' if rms<0.5 else 'MARGINAL'}", flush=True)
