#!/bin/bash
# Paper recipe, CIFAR-10 eps=8/255: gamma_max=4.0, m_max=2.5, lam_d=0.3, guard theta=0.08.
# Reported: AutoAttack 50.45 +- 0.20, CR 50.44 +- 0.16 over 4 seeds (logs/cifar10/guard).
# Usage: bash scripts/train_cifar10.sh [GPU] [SEED ...]   (defaults: 0  0 1 2 3)
set -e
GPU=${1:-0}; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3)
for S in "${SEEDS[@]}"; do
  OUT="results/cifar10/guard_s${S}"
  python train.py --dataset cifar10 --model resnet18 --gpu "$GPU" --seed "$S" \
      --nuc-max 4.0 --mse-extra 2.5 --lam-d 0.3 \
      --guard val --guard-thr 0.08 --out-dir "$OUT"
done
