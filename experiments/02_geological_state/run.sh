#!/usr/bin/env bash
# Sample-level geological state: sparse fine-scan calibration and its alternatives (Sections 4.3-4.4).
set -e
cd "$(dirname "$0")"
GPU=${GPU:-0}
python ana56.py                                   # spatial structure of c and coarse-scan predictors (CPU)
for f in CQ SC YN GZ SD SHX; do python eval56.py $f $GPU; done; python agg56.py   # sparse fine-scan calibration
python chk56.py; python out56b.py; python lvl56.py
for f in CQ SC YN GZ SD SHX; do python eval57.py $f $GPU; done; python agg57.py   # coarse-scan sample level
python lvl57.py
for f in CQ SC YN GZ SD SHX; do python eval57b.py $f $GPU; done; python agg57b.py
python var56.py; python rev56.py; python ana57c.py   # variance decomposition, scale analysis, level-prediction table
python vis_paper.py                               # example blocks used in the figures
