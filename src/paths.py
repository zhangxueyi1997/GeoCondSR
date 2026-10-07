"""Paths used by all scripts.

Set the environment variables below to point to your copies of the data;
by default everything lives inside the repository folder.

    GEOCOND_DATA   paired-block dataset and derived inputs (pairs/, meta/, cfield2/, degradation/, plugs/, ...)
    GEOCOND_RUNS   model checkpoints (one sub-folder per run, e.g. CQ_g54b/)
    GEOCOND_OUT    evaluation outputs
    GEOCOND_RAW    raw whole-plug CT slices (only needed for the whole-plug preprocessing)
    GEOCOND_FIGDATA  aggregated results used by the figure scripts
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('GEOCOND_DATA', REPO_ROOT / 'data'))
RUNS_ROOT = Path(os.environ.get('GEOCOND_RUNS', REPO_ROOT / 'runs'))
OUT_ROOT = Path(os.environ.get('GEOCOND_OUT', REPO_ROOT / 'outputs'))
RAW_ROOT = Path(os.environ.get('GEOCOND_RAW', REPO_ROOT / 'raw'))
FIGDATA = Path(os.environ.get('GEOCOND_FIGDATA', OUT_ROOT / 'figure_data'))

_FOLDS = json.loads((REPO_ROOT / 'configs' / 'folds.json').read_text(encoding='utf-8'))


def fold_meta(fold):
    """Training and test samples of one leave-one-region-out fold (identical for all models)."""
    return _FOLDS[fold]
