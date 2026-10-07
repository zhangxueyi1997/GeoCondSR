#!/usr/bin/env bash
# Comparison with the baselines (Section 4.1, Table 2). The first pass (SKIP59=1) evaluates all methods
# except SwinIR-3D and the diffusion model and stores the binary pore masks used by the flow simulation.
set -e
cd "$(dirname "$0")"
GPU=${GPU:-0}
for f in CQ SC YN GZ SD SHX; do SKIP59=1 python eval59.py $f $GPU; done
for f in CQ SC YN GZ SD SHX; do python eval59.py $f $GPU; done
