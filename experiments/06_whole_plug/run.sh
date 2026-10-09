#!/usr/bin/env bash
# Application to whole core plugs (Sections 3.5 and 4.4).
# Steps 1-4 need the raw whole-plug slices ($GEOCOND_RAW); the released dataset already contains their outputs.
set -e
cd "$(dirname "$0")"
GPU=${GPU:-0}
if [ "${PREPARE:-0}" = "1" ]; then
  python extract_plugs.py      # 1. 300 context windows per plug along the plug height
  python cup_plugs.py          # 2. near-surface air reference and radial cupping correction
  python airnear_plugs.py      # 3. air-reference check
  python dom59.py              # 4. whole-plug vs miniplug power spectra (noise spectrum N(f))
fi
for f in CQ SC YN GZ SD SHX; do
  python noise59.py $f $GPU                                    # noise control on miniplug blocks
  M_SUF=_m59n G_SUF=_n59n python noise59.py $f $GPU
  python plugsr59.py $f $GPU                                   # models without fine-tuning: raw, spectral subtraction, calibrated
  M_SUF=_m59n G_SUF=_n59n python plugsr59.py $f $GPU           # GeoCondSR fine-tuned, baselines not fine-tuned (control)
  M_SUF=_m59n G_SUF=_n59n B_SUF=59n python plugsr59.py $f $GPU # all networks fine-tuned (final whole-plug results)
  python wcsparse59.py $f $GPU                                 # whole-plug workflow with sparse fine-scan calibration
done
B_SUF=59n python wc_vis59.py
