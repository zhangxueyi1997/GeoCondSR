#!/usr/bin/env bash
# Lattice Boltzmann flow simulation on the reconstructed blocks (Section 4.3). Run 03_baselines first.
set -e
cd "$(dirname "$0")"
GPU=${GPU:-0}
for f in CQ SC YN GZ SD SHX; do python lbm_eval59.py $f $GPU eval59a; python lbm_eval59.py $f $GPU eval59; done
python topk59.py; python phi59.py
