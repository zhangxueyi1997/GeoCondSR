"""Re-run one training job of the paper from its recorded configuration.

    python scripts/train_from_config.py CQ_g54b --gpu 0

Each file in configs/runs/ stores the script and the exact arguments of one run
(``${DATA}`` and ``${RUNS}`` are replaced by the paths in src/paths.py).
Runs must be trained in the order given in scripts/train_all.sh, because later
stages start from earlier checkpoints.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from paths import DATA_ROOT, RUNS_ROOT  # noqa: E402


def build_command(run, gpu):
    cfg = json.loads((REPO / 'configs' / 'runs' / (run + '.json')).read_text(encoding='utf-8'))
    cmd = [sys.executable, str(REPO / cfg['script'])]
    if cfg['script'].endswith('/train.py'):
        cmd += ['--root', str(DATA_ROOT)]
    for key, val in cfg['args'].items():
        if val is None or val is False:
            continue
        if val is True:
            cmd.append('--' + key)
            continue
        if isinstance(val, str):
            val = val.replace('${DATA}', str(DATA_ROOT)).replace('${RUNS}', str(RUNS_ROOT))
        cmd += ['--' + key, str(val)]
    cmd += ['--gpu', str(gpu), '--out', str(RUNS_ROOT / run)]
    return cmd


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', help='run name, e.g. CQ_g54b (see configs/runs/)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--dry', action='store_true', help='print the command without running it')
    a = ap.parse_args()
    cmd = build_command(a.run, a.gpu)
    print(' '.join(cmd), flush=True)
    if not a.dry:
        sys.exit(subprocess.call(cmd))
