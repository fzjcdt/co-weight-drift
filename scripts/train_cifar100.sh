#!/bin/bash
# Paper recipe, CIFAR-100 eps=8/255: gamma_max=1.25, m_max=4.5, lam_d=0, guard theta=0.08.
# Reported: AutoAttack 27.15 +- 0.04, CR 27.20 +- 0.07 over 4 seeds (logs/cifar100/guard).
# Usage: bash scripts/train_cifar100.sh [GPU] [SEED ...]   (defaults: 0  0 1 2 3)
set -e
GPU=${1:-0}; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3)
for S in "${SEEDS[@]}"; do
  OUT="results/cifar100/guard_s${S}"
  python train.py --dataset cifar100 --model resnet18 --gpu "$GPU" --seed "$S" \
      --nuc-max 1.25 --mse-extra 4.5 --lam-d 0 \
      --guard val --guard-thr 0.08 --out-dir "$OUT"
done
