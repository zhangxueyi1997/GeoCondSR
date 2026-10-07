# -*- coding: utf-8 -*-
"""第 59 步：退化感知微调（抗成像噪声）。在第 54 步终版权重（_m54b / _n54b 或 _g54b）上继续训练，粗扫输入加入与整柱粗扫同谱的随机噪声：
  噪声谱 N(f) = 本折训练岩性配对组 (P整柱粗扫 − P小柱粗扫) 的中位数（只用两种粗扫，不用细扫；f < 0.15 周/体素置 0），
  每个样本幅度 a ~ U(0, 1.3)；目标仍是干净细扫；训练中的硬投影与退化一致项都对加噪后的粗扫做（与推理一致）。
  阶段一：M 从 --init_ckpt 热启动；阶段二：G、D（及优化器）从 --init_ckpt 热启动，M 用阶段一微调后的权重；判别器输入噪声固定在下限。
其余与第 54 步逐字相同。原说明如下。
第 54 步训练：两阶段。
阶段一（--stage 1）：均值通路 M，纯全图逐体素 L1（学 EDSR 的经验：能从粗扫推出来的，逐体素钉住）。
阶段二（--stage 2）：M 冻结；残差纹理生成器 G（从 _rot2 的 G 热启动）+ 条件判别器 D（看整张图）。
  G 的损失全部对准「真残差 r* = 真值 − μ」：
    · 局部能量对齐（新）：r 与 r* 在 7³ 格上的局部强度逐格对齐 —— 管「纹理该在哪、该多强」，不管具体相位；
      依据：第 54 步前置检验，r* 局部强度与 μ 的边缘强度相关 0.58（上限 0.71），位置可推。
    · 残差谱约束：方向分箱 + 逐壳平坦度 + 径向（第 51 步，抗条纹/斑块）；
    · 结构损失（孔隙度/批级拓扑/局部拓扑）作用在 8 轮硬投影后的整图上（第 42 步口径）；
    · 对抗：D 看整张图 x=μ+r（真值对照），投影式地质条件；梯度只到 G。
  --geo 1：G 与 D 收实测地质 c；--geo 0：同网络同初始化，c 换成训练折均值常数（训练时就无地质信息）。
用法：python train54.py --stage 1 --fold CQ --seed 101 --gpu 0 --out runs/CQ_m54
      python train54.py --stage 2 --geo 1 --fold CQ --seed 101 --gpu 0 --mean_ckpt .../CQ_m54/ckpt/final.pt --init_g .../CQ_rot2/ckpt/step0010000.pt --out ..."""
import sys, json, time, argparse, signal, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, fold_meta  # noqa: E402
from dataset import PairDataset, load_H
from models import degrade, PORE_CUT, chi_soft, thick_soft, chi_local, ARR_SCALE, Discriminator, hinge_d, hinge_g, N_C, Generator
from models54 import MeanPath, MeanPath2, load_mean, load_cpred
from train import spec_loss, spec_dir_loss, proj_hard, _mk_winit
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument('--stage', type=int, required=True); p.add_argument('--fold', required=True)
p.add_argument('--seed', type=int, required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=20000); p.add_argument('--bs', type=int, default=4); p.add_argument('--workers', type=int, default=4)
p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--aug', type=int, default=1); p.add_argument('--resume', default='auto')
p.add_argument('--mver', type=int, default=1, help='阶段一均值通路版本：1=Res3d（第54步初版），2=干净残差块（修正版）')
p.add_argument('--c_src', default='blk', help='第55步：地质变量支撑 blk（含目标块，诊断用）/ ann（只取周围，去循环）')
p.add_argument('--cpred', default='', help='第55步：粗扫→ĉ 预测器检查点；给了就用 ĉ 代替实测 c（工程可用）')
p.add_argument('--mean_ckpt', default=''); p.add_argument('--init_g', default=''); p.add_argument('--geo', type=int, default=1)
p.add_argument('--lr_fa', type=float, default=1e-3, help='新接口（M 特征→G）学习率：零初始化的新参数要比热启动部分学得快（第 52 步教训）')
p.add_argument('--w_en', type=float, default=1.0); p.add_argument('--w_spec', type=float, default=1.0)
p.add_argument('--w_top', type=float, default=2.0); p.add_argument('--w_loc', type=float, default=6.0); p.add_argument('--w_por', type=float, default=10.0)
p.add_argument('--w_deg', type=float, default=1.0); p.add_argument('--proj_train', type=int, default=8)
p.add_argument('--w_adv', type=float, default=0.1); p.add_argument('--lr_d', type=float, default=1e-4); p.add_argument('--d_every', type=int, default=2)
p.add_argument('--d_noise', type=float, default=0.05); p.add_argument('--d_noise_floor', type=float, default=0.02)
p.add_argument('--r1', type=float, default=0.1); p.add_argument('--r1_every', type=int, default=4)
p.add_argument('--init_ckpt', required=True); p.add_argument('--noise_amp', type=float, default=1.3); p.add_argument('--dom', default=str(DATA_ROOT) + '/plugs/dom59.json')
cfg = p.parse_args()
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device('cuda:%d' % cfg.gpu); torch.cuda.set_device(dev)
out = Path(cfg.out); (out / 'ckpt').mkdir(parents=True, exist_ok=True)
STOP = {'f': False}; signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__('f', True)); signal.signal(signal.SIGINT, lambda *a: STOP.__setitem__('f', True))
ROOT = str(DATA_ROOT)
fm = fold_meta(cfg.fold)           # 与既往全部模型同一折划分
ds = PairDataset(ROOT, fm['train'], unit_split='train', aug=bool(cfg.aug), c_src=cfg.c_src)
CP = load_cpred(cfg.cpred, dev) if cfg.cpred else None
g_ = torch.Generator(); g_.manual_seed(cfg.seed)
dl = DataLoader(ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers, drop_last=True, pin_memory=True,
                persistent_workers=True, generator=g_, worker_init_fn=_mk_winit(cfg.seed))
Hk = torch.from_numpy(load_H(ROOT)[0]).to(dev)
scaler = torch.amp.GradScaler('cuda')

M = load_mean(cfg.init_ckpt if cfg.stage == 1 else cfg.mean_ckpt, dev)
if cfg.stage == 1:
    M.train(); [q.requires_grad_(True) for q in M.parameters()]
    optM = torch.optim.AdamW(M.parameters(), lr=cfg.lr, weight_decay=1e-4)
    nets = dict(model=M, opt=optM)
else:
    M.eval(); [q.requires_grad_(False) for q in M.parameters()]
    G = Generator(n_cond=N_C, ch=64, nres=4, nz=8, level=True, see_a=True, norm_a=True, afeat=M.ch).to(dev)
    G.load_state_dict(torch.load(cfg.init_ckpt, map_location=dev, weights_only=False)['model'])
    fa = [q for n, q in G.named_parameters() if n.startswith('fa')]; rest = [q for n, q in G.named_parameters() if not n.startswith('fa')]
    optG = torch.optim.AdamW([dict(params=rest, lr=cfg.lr), dict(params=fa, lr=cfg.lr_fa)], weight_decay=1e-4)
    D = Discriminator().to(dev); optD = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.0, 0.99))
    kk = np.random.default_rng(1).choice(len(ds), min(300, len(ds)), replace=False)
    CCONST = torch.from_numpy(np.stack([ds[int(i)]['c'] for i in kk]).mean(0).astype(np.float32))[None].to(dev)
    nets = dict(model=G, opt=optG, D=D, optD=optD)
    d0 = torch.load(cfg.init_ckpt, map_location=dev, weights_only=False)
    for k_ in ('opt', 'D', 'optD'): nets[k_].load_state_dict(d0[k_])
    del d0
# ---------- 同谱噪声（第 59 步） ----------
_DOM = json.load(open(cfg.dom)); _n = 36; _w1 = np.hanning(_n); _W3 = _w1[:, None, None] * _w1[None, :, None] * _w1[None, None, :]
_f = np.fft.fftfreq(_n); _FR = np.sqrt(_f[:, None, None] ** 2 + _f[None, :, None] ** 2 + _f[None, None, :] ** 2)
_EDG = np.array([0.02, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.87]); _CEN = (_EDG[:-1] + _EDG[1:]) / 2
_TRG = [g for g in fm['train'] if g in _DOM]
NF = np.median([np.array(_DOM[g]['pl']) - np.array(_DOM[g]['ps']) for g in _TRG], 0); NF[_CEN < 0.15] = 0; NF = np.clip(NF, 0, None)
AMP = torch.from_numpy(np.interp(_FR, np.r_[0, _CEN], np.r_[0.0, np.sqrt(NF / ((_W3 ** 2).sum() / _W3.sum()))]).astype(np.float32)).to(dev)
print('噪声谱来自训练组', _TRG, np.round(NF * 1e3, 2), flush=True)


def add_noise(x):
    n = torch.fft.ifftn(torch.fft.fftn(torch.randn_like(x), dim=(-3, -2, -1)) * AMP, dim=(-3, -2, -1)).real
    return x + cfg.noise_amp * torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * n

step = 0
cks = sorted((out / 'ckpt').glob('step*.pt'))
if cfg.resume == 'auto' and cks:
    d = torch.load(cks[-1], map_location=dev, weights_only=False)
    for k, v in nets.items(): v.load_state_dict(d[k])
    step = d['step']; print('从 %s 续跑' % cks[-1].name, flush=True)
meta = dict(fold=cfg.fold, train=fm['train'], test=fm['test'], stage=cfg.stage, cfg=vars(cfg), arch='第59步 退化感知微调（同谱噪声）', NF=NF.tolist(),
            params_M=sum(q.numel() for q in M.parameters()), params_G=(sum(q.numel() for q in G.parameters()) if cfg.stage == 2 else 0))
(out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print('阶段 %d  折 %s  M %.2fM  %s' % (cfg.stage, cfg.fold, meta['params_M'] / 1e6,
      ('G %.2fM  地质 %s' % (meta['params_G'] / 1e6, '实测c' if cfg.geo else '常数c（无地质对照）')) if cfg.stage == 2 else '纯 L1'), flush=True)


def save(s_, final=False):
    tmp = out / 'ckpt' / 'tmp.pt'
    torch.save(dict({k: v.state_dict() for k, v in nets.items()}, step=s_, cfg=vars(cfg)), tmp)
    tmp.rename(out / 'ckpt' / ('final.pt' if final else 'step%07d.pt' % s_))
    for old in sorted((out / 'ckpt').glob('step*.pt'))[:-3]: old.unlink()


def energy(x):   # 7³ 格上的局部强度（112/7 = 16³ 个格）
    return F.avg_pool3d(x.float().pow(2), 7, stride=7)


logf = open(out / 'log.jsonl', 'a'); t0 = time.time(); it = iter(dl)
k = 16; o = (36 - k) // 2; SL = slice(o, o + k)
while step < cfg.steps and not STOP['f']:
    try: b = next(it)
    except StopIteration: it = iter(dl); b = next(it)
    lr_ = b['lr'].to(dev, non_blocking=True); hr_ = b['hr'].to(dev, non_blocking=True).float()
    lr_ = add_noise(lr_.float())
    P = {}
    if cfg.stage == 1:
        with torch.amp.autocast('cuda'):
            mu = M(lr_)
        loss = F.l1_loss(mu.float(), hr_); P['l1'] = float(loss.detach())
        optM.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optM); scaler.update()
    else:
        if not cfg.geo:
            c_ = CCONST.expand(lr_.shape[0], -1)
        elif CP is not None:
            with torch.no_grad(): c_ = CP(lr_).float()          # 只用粗扫推出的 ĉ
        else:
            c_ = b['c'].to(dev, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            mu, feat = M(lr_, ret_feat=True)
        mu = mu.float(); rt = hr_ - mu
        with torch.amp.autocast('cuda'):
            mid, fine, x = G(mu, c_, a_feat=feat)
        r = (mid + fine).float(); x = x.float()
        L = {}
        L['spec'] = spec_dir_loss(r, rt)[0] + spec_loss(r, rt)
        Et, Ep = energy(rt), energy(r); eps = (Et.mean(dim=(1, 2, 3, 4), keepdim=True) * 1e-3).detach()
        L['en'] = (torch.log(Ep + eps) - torch.log(Et + eps)).abs().mean()
        xp = proj_hard(x, lr_, Hk, cfg.proj_train)
        L['top'] = ((chi_soft(xp).mean() - chi_soft(hr_).mean()).abs() / ARR_SCALE[0]
                    + (thick_soft(xp).mean() - thick_soft(hr_).mean()).abs() / ARR_SCALE[2])
        L['loc'] = (chi_local(xp, 4).sort(1).values - chi_local(hr_, 4).sort(1).values).abs().mean() / ARR_SCALE[0]
        L['por'] = (torch.sigmoid((PORE_CUT - xp) / 0.02).mean(dim=(1, 2, 3, 4)) - (hr_ < PORE_CUT).float().mean(dim=(1, 2, 3, 4))).abs().mean()
        L['deg'] = F.l1_loss(degrade(x, Hk), lr_[..., SL, SL, SL])
        sig = cfg.d_noise_floor                                   # 微调：判别器输入噪声保持在原训练末的下限
        nz = lambda t: t + sig * torch.randn_like(t)
        if step % cfg.d_every == 0:
            ld = hinge_d(D(nz(hr_), c_), D(nz(x.detach()), c_))
            if cfg.r1 > 0 and step % (cfg.r1_every * cfg.d_every) == 0:
                rb = nz(hr_).requires_grad_(True); g1, = torch.autograd.grad(D(rb, c_).sum(), rb, create_graph=True)
                ld = ld + 0.5 * cfg.r1 * cfg.r1_every * g1.pow(2).sum(dim=(1, 2, 3, 4)).mean()
            optD.zero_grad(set_to_none=True); ld.backward(); optD.step(); P['d_loss'] = float(ld.detach())
        L['adv'] = hinge_g(D(x, c_))
        loss = (cfg.w_spec * L['spec'] + cfg.w_en * L['en'] + cfg.w_top * L['top'] + cfg.w_loc * L['loc']
                + cfg.w_por * L['por'] + cfg.w_deg * L['deg'] + cfg.w_adv * L['adv'])
        optG.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(optG); scaler.update()
        P.update({k_: float(v.detach()) for k_, v in L.items()})
    step += 1
    if step % 20 == 0:
        P.update(step=step, loss=float(loss.detach()), elapsed_min=(time.time() - t0) / 60, mem_gb=torch.cuda.max_memory_allocated(dev) / 1e9)
        logf.write(json.dumps(P, ensure_ascii=False) + '\n'); logf.flush()
        (out / 'heartbeat.json').write_text(json.dumps(dict(step=step, loss=P['loss'])))
    if step % 500 == 0: save(step)
save(step);
if step >= cfg.steps: save(step, final=True)
print('结束于第 %d 步（%.1f 分钟）' % (step, (time.time() - t0) / 60), flush=True)
