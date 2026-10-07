#!/usr/bin/env bash
# Train every model of the paper for the six leave-one-region-out folds, in dependency order.
# Usage: bash scripts/train_all.sh [gpu]
# Stage names:
#   proj  generator pre-training (30 000 steps)        rot2  rotation augmentation + adversarial (10 000 steps)
#   m54b  mean path                                    n54b / g54b / g54c / g54a / g54n  residual texture generator
#         (geology-free / geology-conditioned / coarse-state / fine-scan statistics of the target neighbourhood / regression on neighbouring fine-scan statistics)
#   cpred geological-state predictor from the coarse scan
#   edsr, srgan, swin, diff   baselines
#   m59n, n59n, g59n          degradation-aware fine-tuning for whole-plug scans
set -e
GPU=${1:-0}
cd "$(dirname "$0")/.."
for FOLD in CQ SC YN GZ SD SHX; do
  for STAGE in proj rot2 m54b n54b g54b g54c g54a g54n cpred edsr srgan swin diff m59n n59n g59n; do
    python scripts/train_from_config.py ${FOLD}_${STAGE} --gpu "$GPU"
  done
done
