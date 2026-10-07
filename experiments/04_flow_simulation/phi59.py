import numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src')); from paths import OUT_ROOT  # noqa: E402
N = 112
for f in ('CQ', 'SC', 'YN', 'GZ', 'SD', 'SHX'):
    D = np.load(str(OUT_ROOT) + '/eval59/out/eval59a_%s_masks.npz' % f)
    phi = {k: np.mean([np.unpackbits(r)[:N ** 3].mean() for r in D[k]]) for k in D.files if k != 'idx'}
    t = phi['细扫']; print(f, '细扫 %.4f |' % t, ' '.join('%s %.4f(%+.0f%%)' % (k, v, 100 * (v / t - 1)) for k, v in phi.items() if k != '细扫'), flush=True)
