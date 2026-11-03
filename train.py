"""Training script: coupled fast adversarial training + EMA self-anchor guard.

Implements the method of the paper
"Catastrophic Overfitting is Weight Drift the Training Loss Cannot See".

Per training step (3 forward / 2 backward passes):
  1. delta0 ~ fresh random half-sign init at 0.5*eps (NuAT-style), z0 = f(x+delta0)
     kept in the graph as the attack point;
  2. one sign step on CE_ls at the attack point gives the adversarial corner
     delta_adv, z_a = f(x+delta_adv); a third clean forward gives z_c = f(x);
  3. coupled loss (Eq. 2 of the paper)
         L = CE_ls(z_c, y) + gamma(t)/B * ||Z_c - Z_a||_* + m(t) * ||z_c - z_a||^2
     where gamma(t), m(t) ramp linearly over training; odd steps additionally
     carry the nuclear term inside the attack objective (NuAT placement).
     For CIFAR-10 an additional small wrong-logit dispersion term (weight
     --lam-d, see paper appendix) is part of the recipe; C100/Tiny use 0.

Guard (--guard val): after each epoch the validation PGD-10 accuracy of the raw
and EMA weights is checked against the running best; a drop larger than
--guard-thr recenters the raw weights on their own EMA (the pre-collapse
basin). Cooldown --guard-cd and budget --guard-max make the rule inert on
healthy trajectories (it never fires on 10 of our 13 reported runs).

Protocol follows FastAT_Benchmark: ResNet18 (PreActResNet18 on Tiny-ImageNet),
100 epochs, OneCycleLR, SGD(0.1, momentum .9, wd 5e-4), batch 128, 49000/1000
train/val split, label smoothing 0.4, EMA tau .9995. best.pth = epoch with the
highest val PGD-10 (raw vs EMA, tie-broken by clean acc).

Outputs per run dir: train.log (per-epoch line), probe.csv (optional batch
probes: clean/FGSM/PGD-10 acc + grad cosine), best.pth / last.pth / ema_last.pth.
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchattacks import PGD

from models import ResNet18, PreActResNet18
from utils.dataset import (train_transform, test_transform,
                           train_transform_tiny_imagenet, TinyImageNet200)
from utils.set_seed import set_seed


def probe_batch(model, px, py, eps, device):
    """Cheap batch probes every k steps: (clean, fgsm, pgd10, gcos) on fixed 256."""
    import torch.nn.functional as _F
    was_training = model.training
    model.eval()
    with torch.no_grad():
        clean = (model(px).argmax(1) == py).float().mean().item()
    x = px.clone().detach().requires_grad_(True)
    z = model(x)
    g1, = torch.autograd.grad(_F.cross_entropy(z, py, label_smoothing=0.4), x)
    with torch.no_grad():
        fgsm = (model((px + eps * g1.sign()).clamp(0, 1)).argmax(1) == py).float().mean().item()
    delta = torch.zeros_like(px).uniform_(-eps, eps).detach().requires_grad_(True)
    for _ in range(10):
        za = model((px + delta).clamp(0, 1))
        ga, = torch.autograd.grad(_F.cross_entropy(za, py, label_smoothing=0.4), delta)
        delta.data = (delta + 2.0 / 255 * ga.sign()).clamp(-eps, eps)
    with torch.no_grad():
        pgd = (model((px + delta).clamp(0, 1)).argmax(1) == py).float().mean().item()
    d2 = delta.detach().clone().requires_grad_(True)
    za2 = model((px + d2).clamp(0, 1))
    g3, = torch.autograd.grad(_F.cross_entropy(za2, py, label_smoothing=0.4), d2)
    gcos = F.cosine_similarity(g1.flatten(1), g3.flatten(1), 1).mean().item()
    if was_training:
        model.train()
    return clean, fgsm, pgd, gcos


def nuc(G, tail=0):
    """Exact differentiable nuclear norm via CPU LAPACK SVD (straight-through
    subgradient). P = U V^T is the exact nuclear-norm subgradient and
    <P, G> = sum(sigma) exactly. CPU LAPACK never hits the cuSOLVER
    convergence failures we see on B x 100+ logit matrices, and the backward
    has no 1/(eigengap) blowup because only G carries gradient.
    tail=k > 0: tail-only nuclear norm sum_{i>k} sigma_i (top-k singular
    directions exempted from the penalty)."""
    with torch.no_grad():
        U, _, Vh = torch.linalg.svd(G.detach().float().cpu(), full_matrices=False)
        P = (U @ Vh).to(device=G.device, dtype=G.dtype)
        if tail > 0:
            P = P - (U[:, :tail] @ Vh[:tail, :]).to(device=G.device, dtype=G.dtype)
    return (P * G).sum()


def wrong_disp(z, y):
    """Mean over batch of Var_{j!=y}(z_j) (wrong-logit dispersion)."""
    onehot = F.one_hot(y, z.size(1)).float()
    m = (z * (1 - onehot)).sum(1) / (z.size(1) - 1)
    var = (((z - m[:, None]) ** 2) * (1 - onehot)).sum(1) / (z.size(1) - 1)
    return var.mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--eps', type=float, default=8.0 / 255)
    ap.add_argument('--nuc-max', type=float, default=4.0,   # gamma_max: nuclear ramp target
                    help='gamma_max; per-dataset values: 4.0 (C10) / 1.25 (C100) / 1.0 (Tiny)')
    ap.add_argument('--mse-extra', type=float, default=2.5,  # m_max: MSE ramp target
                    help='m_max; per-dataset values: 2.5 (C10) / 4.5 (C100) / 2.5 (Tiny)')
    ap.add_argument('--lam-d', type=float, default=0.3,
                    help='wrong-logit dispersion weight (CIFAR-10 recipe: 0.3; C100/Tiny: 0)')
    ap.add_argument('--mse-delay', type=int, default=0)     # delay MSE ramp start by N epochs
    ap.add_argument('--ls', type=float, default=0.4)        # label smoothing dose
    ap.add_argument('--nuc-tail', type=int, default=0,
                    help='exempt top-k singular values from the nuclear penalty')
    ap.add_argument('--save-every', type=int, default=0)    # save ckpt_e{N}.pth every N epochs
    ap.add_argument('--probe-every', type=int, default=50,  # batch probes (0=off)
                    help='diagnostic batch probes every k steps; NOTE: consuming '
                         'RNG here perturbs the training trajectory (lottery effect)')
    ap.add_argument('--guard', default='val', choices=['none', 'val'],
                    help="'val': EMA self-anchor guard on (the paper's method); 'none': base only")
    ap.add_argument('--guard-thr', type=float, default=0.08)  # val PGD drop threshold
    ap.add_argument('--guard-cd', type=int, default=3)        # cooldown epochs
    ap.add_argument('--guard-max', type=int, default=5)       # max fires (-1 = unlimited)
    ap.add_argument('--init-style', default='halfsign', choices=['halfsign', 'bu'],
                    help='halfsign: fresh half-sign init at 0.5*eps (paper); '
                         'bu: 70%% +-eps Rademacher / 30%% U[-eps,eps]')
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'])
    ap.add_argument('--model', default='resnet18', choices=['resnet18', 'preact18'])
    ap.add_argument('--data-root', default='./data',
                    help='root holding cifar-10-batches-py / cifar-100-python / tiny-imagenet-200.zip')
    ap.add_argument('--out-dir', default=None,
                    help='run dir (default results/<dataset>/<tag>_s<seed>)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    set_seed(args.seed)
    eps = args.eps
    ls = args.ls
    tau = 0.9995

    NC = {'cifar10': 10, 'cifar100': 100, 'tiny-imagenet': 200}[args.dataset]
    if args.dataset == 'cifar10':
        train_sub = torch.utils.data.Subset(
            datasets.CIFAR10(args.data_root, train=True, download=True, transform=train_transform),
            np.arange(49000))
        val_sub = torch.utils.data.Subset(
            datasets.CIFAR10(args.data_root, train=True, download=True, transform=test_transform),
            np.arange(49000, 50000))
    elif args.dataset == 'cifar100':
        train_sub = torch.utils.data.Subset(
            datasets.CIFAR100(args.data_root, train=True, download=True, transform=train_transform),
            np.arange(49000))
        val_sub = torch.utils.data.Subset(
            datasets.CIFAR100(args.data_root, train=True, download=True, transform=test_transform),
            np.arange(49000, 50000))
    else:
        train_sub = TinyImageNet200(args.data_root, train=True, download=True,
                                    transform=train_transform_tiny_imagenet)
        val_sub = TinyImageNet200(args.data_root, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_sub, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_sub, batch_size=500, shuffle=False, num_workers=2)

    model = (ResNet18(num_classes=NC) if args.model == 'resnet18'
             else PreActResNet18(num_classes=NC)).to(device)
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = OneCycleLR(opt, max_lr=0.1, total_steps=args.epochs)
    wa_model = copy.deepcopy(model)
    exp_avg = copy.deepcopy(model.state_dict())

    outdir = args.out_dir or f'results/{args.dataset}/{args.tag}_s{args.seed}'
    os.makedirs(outdir, exist_ok=True)
    log = open(f'{outdir}/train.log', 'w')
    P = lambda s: (print(s, flush=True), log.write(s + '\n'), log.flush())
    probef = open(f'{outdir}/probe.csv', 'w')
    probef.write('gbatch,epoch,clean,fgsm,pgd10,gcos\n')
    probe_px = probe_py = None
    P(f'train.py seed={args.seed} epochs={args.epochs} dataset={args.dataset} model={args.model} '
      f'nuc_max={args.nuc_max} mse_extra={args.mse_extra} lam_d={args.lam_d} ls={args.ls} '
      f'init={args.init_style} guard={args.guard}/{args.guard_thr} eps={args.eps}')

    ce_ls = lambda z, y: F.cross_entropy(z, y, label_smoothing=ls, reduction='none')
    best_pgd, best_val, best_src = 0.0, 0.0, 'model'
    step = 0
    adv_acc_sum, adv_acc_n = 0.0, 0
    fires, last_fire_ep = 0, -10

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            B = images.size(0)
            onehot = F.one_hot(labels, NC).float()

            # ---- init delta0 ----
            if args.init_style == 'halfsign':
                # nu_at-verbatim fresh half-sign init at 0.5 eps
                delta0 = eps * 0.5 * torch.sign(
                    torch.tensor([0.5]).to(device) - torch.rand_like(images))
            else:
                # 70% +-eps Rademacher / 30% U[-eps,eps]
                r = torch.rand_like(images)
                sgn = torch.randint(0, 2, images.shape, device=device) * 2 - 1
                unif = (2 * torch.rand_like(images) - 1) * eps
                delta0 = torch.where(r < 0.7, eps * sgn, unif)

            if args.probe_every and step % args.probe_every == 0:
                if probe_px is None:
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            probe_px, probe_py = vx[:256].to(device), vy[:256].to(device)
                            break
                pc, pf, pp, gc = probe_batch(model, probe_px, probe_py, eps, device)
                probef.write(f'{step},{epoch},{pc:.4f},{pf:.4f},{pp:.4f},{gc:.4f}\n')
                probef.flush()
            adv0 = (images + delta0).clamp(0, 1).detach().requires_grad_(True)
            z0 = model(adv0)
            z_clean = model(images)  # third (clean) forward

            # ---- attack backward at init point (z0 stays in graph: anchor) ----
            nuc_reg_t = args.nuc_max * epoch / args.epochs
            atk = ce_ls(z0, labels).mean()
            if step % 2 == 1:  # nu_at-verbatim odd-step attack-side nuc
                atk = atk + nuc_reg_t * nuc(z_clean - z0) / B
            opt.zero_grad()
            atk.backward(retain_graph=True)
            d = adv0.grad.detach().sign()
            delta_adv = (delta0 + eps * d).clamp(-eps, eps)
            z_adv = model((images + delta_adv).clamp(0, 1).detach())

            # ---- training loss (Eq. 2): CE at clean + ramped nuclear coupling
            # (batch scale) + ramped MSE coupling (per example), clean <-> adv ----
            loss = (ce_ls(z_clean, labels).mean()
                    + nuc_reg_t * nuc(z_clean - z_adv, tail=args.nuc_tail) / B)
            em_w = args.mse_extra * min(1.0, max(0.0, epoch - args.mse_delay)
                                        / max(1, 0.5 * args.epochs))
            loss = loss + em_w * F.mse_loss(z_clean, z_adv)
            if args.lam_d > 0:   # CIFAR-10 recipe only (9 = NC-1 there); 0 on C100/Tiny
                wmn = (z_adv * (1 - onehot)).sum(1, keepdim=True) / 9.0
                dispn = (((z_adv - wmn) ** 2) * (1 - onehot)).sum(1) / 9.0
                loss = loss + args.lam_d * dispn.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                adv_acc_sum += (z_adv.argmax(1) == labels).float().mean().item()
            adv_acc_n += 1
            step += 1
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    exp_avg[k] = (1 - tau) * v + tau * exp_avg[k]
        sched.step()

        # ---- validation: PGD-10 on raw and EMA + dispersion diagnostics ----
        model.eval()

        def val(m, want_disp=False):
            pgd = PGD(m, eps=eps, alpha=2.0 / 255, steps=10)
            va = pgd_acc = n = 0
            dc = da = 0.0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    zc = m(x)
                va += (zc.argmax(1) == y).sum().item()
                xm = pgd(x, y)
                with torch.no_grad():
                    zm = m(xm)
                pgd_acc += (zm.argmax(1) == y).sum().item()
                if want_disp:
                    dc += wrong_disp(zc, y) * y.size(0)
                    da += wrong_disp(zm, y) * y.size(0)
                n += y.size(0)
            return va / n, pgd_acc / n, dc / n, da / n

        va_m, pgd_m, disp_c, disp_a = val(model, want_disp=True)
        wa_model.load_state_dict(exp_avg)
        wa_model.eval()
        va_w, pgd_w, _, _ = val(wa_model)
        if (pgd_m, va_m) >= (pgd_w, va_w):
            cur, src, cur_va = pgd_m, 'model', va_m
        else:
            cur, src, cur_va = pgd_w, 'wa', va_w
        # ---- guard: self-anchoring on detected CO collapse ----
        adv_ep_mean = adv_acc_sum / max(1, adv_acc_n)
        adv_acc_sum, adv_acc_n = 0.0, 0
        g_cur, g_best = max(pgd_m, pgd_w), best_pgd
        if (args.guard == 'val' and epoch >= 5
                and (args.guard_max < 0 or fires < args.guard_max)
                and epoch - last_fire_ep >= args.guard_cd
                and g_best - g_cur > args.guard_thr):
            fires += 1
            last_fire_ep = epoch
            model.load_state_dict(exp_avg)   # recenter raw on its own EMA (pre-CO basin)
            P(f'GUARD FIRE ep{epoch}: pgd_m {pgd_m:.3f} pgd_w {pgd_w:.3f} best {best_pgd:.3f} '
              f'-> recentred on EMA (fire #{fires}, adv_ep_mean {adv_ep_mean:.3f})')
            wa_model.load_state_dict(exp_avg)
            wa_model.eval()
            va_w2, pgd_w2, _, _ = val(wa_model)
            va_m2, pgd_m2, _, _ = val(model)
            pgd_w, va_w, pgd_m, va_m = pgd_w2, va_w2, pgd_m2, va_m2
            if (pgd_m, va_m) >= (pgd_w, va_w):
                cur, src, cur_va = pgd_m, 'model', va_m
            else:
                cur, src, cur_va = pgd_w, 'wa', va_w
        if cur > best_pgd or (cur == best_pgd and cur_va > best_val):
            best_pgd, best_val, best_src = cur, cur_va, src
            torch.save((model if src == 'model' else wa_model).state_dict(),
                       f'{outdir}/best.pth')
        torch.save(model.state_dict(), f'{outdir}/last.pth')
        torch.save(exp_avg, f'{outdir}/ema_last.pth')
        if args.save_every and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), f'{outdir}/ckpt_e{epoch + 1}.pth')
            torch.save(exp_avg, f'{outdir}/ckpt_e{epoch + 1}_wa.pth')
        P(f'epoch {epoch:3d}  raw_pgd {pgd_m:.4f}  wa_pgd {pgd_w:.4f}  '
          f'best {best_pgd:.4f}({best_src})  disp_c {disp_c:.3f} disp_adv {disp_a:.3f}  '
          f'({time.time() - t0:.0f}s)')
    P('DONE')


if __name__ == '__main__':
    main()
