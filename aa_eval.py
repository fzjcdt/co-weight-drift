"""AutoAttack evaluation (full test set, bs=200 — FastAT_Benchmark protocol).

--full  : complete standard AutoAttack suite (apgd-ce, apgd-t, fab-t, square).
          This is the variant the released numbers in the paper use; it writes
          autoattack_full.log.
default: the benchmark's 2-attack 'customer' variant (apgd-ce + apgd-t);
          faster development eval, differs from the full suite by at most
          0.02 on our models.

Usage:
  python aa_eval.py --run-dir results/cifar10/run_s0 --gpu 0 --full
  python aa_eval.py --run-dir results/tiny-imagenet/run_s0 --gpu 0 --model preact18 --full
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from autoattack import AutoAttack
from models import ResNet18, PreActResNet18
from utils.dataset import test_transform, TinyImageNet200
from utils.set_seed import set_seed

MODELS = {'resnet18': ResNet18, 'preact18': PreActResNet18}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--which', default='best', choices=['best', 'last'])
    ap.add_argument('--eps', type=float, default=8.0 / 255)
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--dataset', default=None, choices=[None, 'cifar10', 'cifar100', 'tiny-imagenet'])
    ap.add_argument('--model', default='resnet18', choices=sorted(MODELS.keys()))
    ap.add_argument('--num-classes', type=int, default=None)
    ap.add_argument('--full', action='store_true',
                    help='complete standard AutoAttack (apgd-ce, apgd-t, '
                         'fab-t, square) — the paper\'s reported numbers')
    args = ap.parse_args()

    # infer dataset from run-dir path results/<dataset>/...
    if args.dataset is None:
        parts = os.path.normpath(args.run_dir).split(os.sep)
        for cand in ['cifar10', 'cifar100', 'tiny-imagenet']:
            if cand in parts:
                args.dataset = cand
                break
    if args.dataset is None:
        args.dataset = 'cifar10'

    ncls = args.num_classes or {'cifar10': 10, 'cifar100': 100, 'tiny-imagenet': 200}[args.dataset]

    device = torch.device(f'cuda:{args.gpu}')
    if args.dataset == 'cifar10':
        DS = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        DS = datasets.CIFAR100
    else:
        DS = TinyImageNet200
    loader = DataLoader(DS(args.data_root, train=False, transform=test_transform, download=True),
                        batch_size=2000, shuffle=False, num_workers=0)
    # FULL test set (a single next(iter(loader)) would truncate tiny-imagenet
    # to its first 2000 images; the benchmark evaluates all 10000).
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    X, y = torch.cat(xs), torch.cat(ys)
    X, y = X.to(device), y.to(device)

    set_seed(0)
    model = MODELS[args.model](num_classes=ncls).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run_dir, f'{args.which}.pth'), map_location=device))
    model.eval()

    log_path = os.path.join(args.run_dir,
                            'autoattack_full.log' if args.full else 'autoattack.log')
    if args.full:
        attacker = AutoAttack(model, eps=args.eps, version='standard', device=device,
                              log_path=log_path)
    else:
        attacker = AutoAttack(model, eps=args.eps, version='customer',
                              attacks_to_run=['apgd-ce', 'apgd-t'], device=device, log_path=log_path)
    attacker.run_standard_evaluation(X, y, bs=200)


if __name__ == '__main__':
    main()
