#!/bin/bash
# Evaluate a run dir with the complete standard AutoAttack + the CR attack
# (both on the full test set, bs=200 — the paper's reported protocol).
# Usage: bash scripts/eval.sh <run-dir> [GPU] [MODEL]
#   MODEL: resnet18 (default) or preact18 (tiny-imagenet)
set -e
DIR=$1; GPU=${2:-0}; MODEL=${3:-resnet18}
python aa_eval.py --run-dir "$DIR" --gpu "$GPU" --model "$MODEL" --full
python cr_eval.py --run-dir "$DIR" --gpu "$GPU" --model "$MODEL"
echo "== $DIR"
tail -1 "$DIR/autoattack_full.log"
grep 'Robust accuracy' "$DIR/cr_attack.log" | tail -1
