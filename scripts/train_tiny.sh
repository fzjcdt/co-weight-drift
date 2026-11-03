#!/bin/bash
# Paper recipe, Tiny-ImageNet eps=8/255 (PreActResNet18):
#   gamma_max=1.0, m_max=2.5, lam_d=0, guard theta=0.08.
# Reported: AutoAttack 20.47 +- 0.11, CR 20.54 +- 0.15 over 3 seeds
# (logs/tiny-imagenet/guard). NOTE: probes off (--probe-every 0) matches the
# released Tiny runs; leaving them on is fine but perturbs the RNG stream.
# Usage: bash scripts/train_tiny.sh [GPU] [SEED ...]   (defaults: 0  0 1 2)
set -e
GPU=${1:-0}; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
for S in "${SEEDS[@]}"; do
  OUT="results/tiny-imagenet/guard_s${S}"
  python train.py --dataset tiny-imagenet --model preact18 --gpu "$GPU" --seed "$S" \
      --nuc-max 1.0 --mse-extra 2.5 --lam-d 0 --probe-every 0 \
      --guard val --guard-thr 0.08 --out-dir "$OUT"
done
