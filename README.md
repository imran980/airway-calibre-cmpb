# Paediatric airway calibre from bronchoscopy video: code and result files

Code and result files for our manuscript on measuring paediatric airway calibre from bronchoscopy video, validated against CT (Computer Methods and Programs in Biomedicine, in preparation). The table under Contents maps each item in the paper to the scripts and result files behind it. The clinical data are not included.

## Check the reported numbers

```
python check_paper_numbers.py
```

This recomputes the numbers in the CT accuracy and ablation tables (baseline and gated estimator, including bias, limits of agreement, Pearson *r*, per-patient Spearman rho and the self-consistency ratios), the 20-V1 CT-reference sensitivity rows and carina shift, the yield totals and the C3VD summary from the JSON files in `runs/`, and compares each with the manuscript; it exits with status 1 on any mismatch. Capture statistics are printed but not checked. Only NumPy and SciPy are needed.

## Contents

| Paper item | File(s) |
|---|---|
| Camera calibration (OPENCV model, k3 = 0, 14x13 inner-corner board) | `experiments/cmig_paper/calibrate_case.py`, `calibrate_case_robust.py` (forced board for defocused clips), `pipeline/calibrate.py` (same method as `calibrate_case_robust.py` for a single video, with command-line options); results `runs/retriage_first15/_calib/*/intrinsics_pinned.json`, `runs/own_data/retry/*_intrinsics.json` |
| Calibration table (Table S1) | `experiments/cmig_paper/calib_table.py` -> `tables/calib_table.tex`, `tables/calib_table_note.tex` |
| Window selection (contact sheets) | `pipeline/contact_sheet.py` |
| Reconstruction: frame extraction, field mask, CLAHE, pinned-intrinsics SfM, model merging, MVS, fusion | `pipeline/recover_clip.py` (post-steps `pipeline/csa_run.py`, `pipeline/render_airway.py`) |
| Measurement: outlier removal, centreline, cross-section slices, coverage gate, three area estimators | `pipeline/airway_analysis.py` |
| Partial-arc calibre estimator (gated estimator) | `pipeline/csa_partialarc.py` |
| Yield and tiers (Table S10, measurability counts) | `experiments/cmig_paper/gallery_v2.py` -> `runs/own_data/reports/yield_table.json`, `tables/yield_table.tex` |
| Capture statistics (Table S5) | `experiments/cmig_paper/capture_analysis.py` -> `runs/own_data/reports/capture_table.json`, `tables/capture_table.tex` |
| Capture mechanism coding (Table S6) | `tables/capture_notes.tex` (coded by review of the frames) |
| Scale-free CSA and % obstruction (Table S7) | `experiments/cmig_paper/robust_csa.py`, `build_csa_report.py` -> `runs/own_data/reports/csa_summary.json`, `tables/obstruction_table.tex` |
| CT reference (TotalSegmentator trachea) | `pipeline/ct_ground_truth.py`, `experiments/cmig_paper/ct_gt_ts.py` |
| CT reference by fixed -900 HU threshold; 20-V1 extraction record (Table S3) | `experiments/cmig_paper/ct_raw_lumen.py` -> `runs/own_data/reports/rawlumen_{INSP,EXP}.json`; `ct_phase_pair_raw.py` -> `ct_phase_pair_20-V1.json` |
| CT accuracy, all-slice baseline | `experiments/cmig_paper/ct3_report.py` -> `runs/own_data/ct_validated_3/CT_validation_metrics.json`, `runs/own_data/reports/ct3_pooled_stats.json` |
| CT accuracy, gated estimator and ablation (accuracy and ablation tables, Table S8) | `experiments/cmig_paper/ct3_regate.py` -> `ct3_regate.json`; `ct3_rerun_eval.py` -> `rerun_eval_*.json`; `ct3_report_v2.py` -> `ct3_pooled_stats_v2.json`; `cam_rings.py` (stations behind the reported 2-V2 registration) |
| 20-V1 CT-reference sensitivity (Table S8) | `experiments/cmig_paper/ct_reference_sensitivity.py` -> `runs/own_data/reports/ct_reference_sensitivity_20-V1.json` |
| C3VD external validation (Table S9) | `experiments/cmig_paper/c3vd_undistort.py`, `c3vd_eval.py`; summary `runs/own_data/renders/new/CAT5_external_validation.json`; intrinsics `runs/own_data/c3vd_intrinsics.json` |
| Manuscript tables as typeset | `tables/*.tex` |
| Figure style | `experiments/cmig_paper/figstyle.py` |

## Data not included

- The bronchoscopy videos, calibration videos, CT scans, extracted frames, point clouds and full CT airway profiles are clinical data and are not distributed. The result files are reduced to the matched reconstruction/CT calibre values at the accepted stations, per-case summaries and the extents of the 20-V1 CT extraction.
- Scripts that read a video, a reconstruction or a CT expect them at the `runs/` paths used in the paper. Set `BRONCHO_DATA` to the folder holding the videos.
- C3VD (Colonoscopy 3D Video Dataset) is public but not redistributed here.

## De-identification

Clip identifiers (e.g. `2-V2`, `20-V1`) are internal study codes. In the calibration and result files, absolute paths to the clinical data were replaced by the clip name and recording dates were removed from video names. Full CT airway profiles were dropped from the result files; the matched station values used for the reported statistics were kept.

## Environment

- Python 3.11 with `requirements.txt` (NumPy 1.26.4, SciPy 1.17.1, OpenCV 4.13.0, Open3D 0.18.0, pycolmap 4.0.4).
- COLMAP 3.13.0 built with CUDA; set `BRONCHO_COLMAP=/path/to/colmap`.
- TotalSegmentator 2.18.0 (CT reference only), in its own environment with `numpy<2`; set `TOTALSEG_BIN` and `TOTALSEG_HOME_DIR`. Run the full task; the `--roi_subset` shortcut under-segments.

## Running the pipeline on a new examination

```
export BRONCHO_COLMAP=/path/to/colmap BRONCHO_DATA=/path/to/videos
python experiments/cmig_paper/calibrate_case_robust.py <case>           # -> runs/own_data/retry/<case>_intrinsics.json
python pipeline/contact_sheet.py --video <clip.mp4> --out sheet.png     # pick a stretch where the scope moves steadily down a centred lumen
python pipeline/recover_clip.py --video <clip.mp4> --calib runs/own_data/retry/<case>_intrinsics.json \
       --lo <first frame> --hi <last frame> --out runs/own_data/retry/rr_<case> --gpus 0,1
python experiments/cmig_paper/ct3_rerun_eval.py <case> runs/own_data/retry/rr_<case>   # CT-paired cases only
```

Frame indices are sequential-decode indices. The windows used for each examination are in `runs/own_data/reports/capture_table.json` (`seg_lo`, `seg_hi`).

## Fixed parameters

| Stage | Setting |
|---|---|
| Field mask | pixels bright in >= 20% of frames; 5x5 opening; 7x7 erosion, 5 iterations (about 15 px) |
| Contrast | CLAHE, clip limit 3.0 |
| Features and matching | SIFT, `max_image_size` 1600, `max_num_features` 8192, `peak_threshold` 0.005; exhaustive matching |
| Mapper | intrinsics pinned (`ba_refine_focal_length 0`, `ba_refine_principal_point 0`, `ba_refine_extra_params 0`); `init_min_tri_angle` 4; `min_num_matches` 8 |
| Dense stereo | undistortion at 1600 px; PatchMatch `geom_consistency 1`, `max_image_size 1200`; other settings COLMAP 3.13 defaults |
| Fusion | geometric input, `min_num_pixels 3` |
| Outlier removal (calibre and partial-arc path) | radial pre-trim at median + 4 MAD; statistical filter *k* 24, std ratio 1.8; radius filter *r* = 0.03 x median point-to-centre distance, >= 16 points |
| Outlier removal (camera-centreline path) | statistical filter *k* 20, std ratio 2.0 |
| Slice gate (CT-paired calibre) | camera-centreline estimator: angular coverage >= 0.75, radius within 3.5 MAD of the median; partial-arc estimator: coverage >= 0.75, circle-fit residual < 0.15, radius within 4 MAD and 0.4-2.5x the median |
| Self-consistency rule | partial-arc and camera-centreline calibres of the same cloud agree within 30% |
| CT registration | isotropic Sim(3): one scale and orientation; scale constrained so the median reconstructed calibre maps into 0.5-2x the CT calibre range; >= 60% of stations matched |

## Licence and funding

MIT licence (see `LICENSE`), copyright the GENESIS project, which also funded the work.
