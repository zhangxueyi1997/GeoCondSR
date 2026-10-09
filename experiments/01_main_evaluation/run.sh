#!/usr/bin/env bash
# Evaluation of the geological-state sources (Section 4.2, Supplementary Table S2).
# C_MODE selects how the geological state of the geology-conditioned generator is obtained:
#   pred  predicted from the coarse scan                       (model _g54c)
#   nbr   regression on neighbouring fine-scan statistics      (model _g54n; needs nbr55.py first)
#   ann   fine-scan statistics around the target block         (model _g54a)
# Results are written to $GEOCOND_OUT/eval/eval55c, eval55n and eval55a.
set -e
cd "$(dirname "$0")"
OUT=${GEOCOND_OUT:-../../outputs}; GPU=${GPU:-0}
python nbr55.py
for MODE in pred nbr ann; do
  case $MODE in pred) G=_g54c; D=eval55c ;; nbr) G=_g54n; D=eval55n ;; ann) G=_g54a; D=eval55a ;; esac
  export EVAL_OUT=$OUT/eval/$D/ C_MODE=$MODE M_SUF=_m54b ARM_G=$G ARM_N=_n54b PREV_G=_g54b PREV_M=_m54b
  mkdir -p "$EVAL_OUT"
  for f in CQ SC YN GZ SD SHX; do python eval55.py $f $GPU; done
  python agg55.py
done
