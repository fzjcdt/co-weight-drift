"""CR Attack (constraint relaxation) evaluation, mirroring FastAT_Benchmark's
cr_test.py exactly: best.pth, full test set, CRAttack with package defaults
(max_iter=150, decay_steps=30, target_numbers=3, restart=5, random_start=True,
seed=0), bs=200, log line 'Robust accuracy: XX.XX%' via utils.Logger.

Usage:
  python cr_eval.py --run-dir results/cifar10/run_s0 --gpu 0
  python cr_eval.py --run-dir results/tiny-imagenet/run_s0 --gpu 0 --model preact18
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# NOTE: utils must be imported before attacks (attacks/attacks/cr_attack does
# `from utils import Logger`, and utils/__init__ imports dataset in turn;
# importing attacks first hits that circular import).
from utils.dataset import test_transform, TinyImageNet200  # noqa: E402
from utils.set_seed import set_seed  # noqa: E402
from attacks import CRAttack  # noqa: E402
from models import ResNet18, PreActResNet18  # noqa: E402

MODELS = {'resnet18': ResNet18, 'preact18': PreActResNet18}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--which', default='best', choices=['best', 'last'])
    ap.add_argument('--eps', type=float, default=8.0 / 255)
    ap.add_argument('--data-root', default='./data')
    ap.add_argument('--dataset', default=None,
                    choices=[None, 'cifar10', 'cifar100', 'tiny-imagenet'])
    ap.add_argument('--model', default='resnet18', choices=sorted(MODELS.keys()))
    ap.add_argument('--num-classes', type=int, default=None)
    ap.add_argument('--log-name', default='cr_attack.log')
    args = ap.parse_args()

    if args.dataset is None:
        parts = os.path.normpath(args.run_dir).split(os.sep)
        for cand in ['cifar10', 'cifar100', 'tiny-imagenet']:
            if cand in parts:
                args.dataset = cand
                break
    if args.dataset is None:
        args.dataset = 'cifar10'

    ncls = args.num_classes or {'cifar10': 10, 'cifar100': 100,
                                'tiny-imagenet': 200}[args.dataset]

    device = torch.device(f'cuda:{args.gpu}')
    if args.dataset == 'cifar10':
        DS = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        DS = datasets.CIFAR100
    else:
        DS = TinyImageNet200
    loader = DataLoader(DS(args.data_root, train=False, transform=test_transform,
                           download=True),
                        batch_size=2000, shuffle=False, num_workers=0)
    # FULL test set (all 10000 images; see aa_eval.py).
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    X, y = torch.cat(xs), torch.cat(ys)

    set_seed(0)
    model = MODELS[args.model](num_classes=ncls).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run_dir, f'{args.which}.pth'),
                                     map_location=device))
    model.eval()

    log_path = os.path.join(args.run_dir, args.log_name)
    attacker = CRAttack(model, eps=args.eps, log_path=log_path)
    attacker.run_standard_evaluation(X, y, bs=200)


if __name__ == '__main__':
    main()
