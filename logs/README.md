# Released run logs

Raw logs of every run behind the paper's tables (no checkpoints — each
`best.pth` is ~45 MB; regenerate with `scripts/train_*.sh`). Sizes are small
enough to keep the full training trace of each run in the repo.

Layout: `logs/<dataset>/<setting>/s<seed>/` with

| file | content |
|---|---|
| `train.log` | per-epoch val PGD-10 (raw + EMA), running best, wrong-logit dispersion diagnostics, guard-fire lines |
| `probe.csv` | batch-level probes (`gbatch,epoch,clean,fgsm,pgd10,gcos`) — the CO dynamics of the paper's figures |
| `autoattack_full.log` | complete standard AutoAttack (apgd-ce, apgd-t, FAB-T, square), full test set |
| `cr_attack.log` | CR attack (constraint relaxation), full test set |
| `autoattack_2attack.log` | 2-attack AutoAttack variant (apgd-ce + apgd-t) — only for the two collapsed base seeds |

## Mapping to the paper

**CIFAR-10, ε=8/255, ResNet18**

| dir | paper run | γ_max | m_max | λ_d | guard | probes | AA | CR |
|---|---|---|---|---|---|---|---|---|
| `cifar10/guard/s0` | Base+guard seed 0 | 4.0 | 2.5 | 0.3 | θ=0.08 | on | 50.40 | 50.40 |
| `cifar10/guard/s1` | Base+guard seed 1 | 4.0 | 2.5 | 0.3 | θ=0.08 | on | 50.19 | 50.23 |
| `cifar10/guard/s2` | Base+guard seed 2 | 4.0 | 2.5 | 0.3 | θ=0.08, fires at ep 19/22/25/98 | off | 50.61 | 50.52 |
| `cifar10/guard/s3` | Base+guard seed 3 | 4.0 | 2.5 | 0.3 | θ=0.08 | off | 50.60 | 50.60 |
| **mean ± std** | Table 1 | | | | | | **50.45 ± 0.20** | **50.44 ± 0.16** |
| `cifar10/base/s0` | Base (no guard), survivor | 4.0 | 2.5 | 0.3 | none | off | 50.72 | — |
| `cifar10/base/s3` | Base (no guard), survivor | 4.0 | 2.5 | 0.3 | none | off | 50.61 | — |
| `cifar10/base/s1` | Base (no guard), **CO-collapsed** | 4.0 | 2.5 | 0.3 | none | off | 7.57* | — |
| `cifar10/base/s2` | Base (no guard), **CO-collapsed** | 4.0 | 2.5 | 0.3 | none | off | 7.48* | — |

\* 2-attack AutoAttack variant; the collapsed checkpoints are gradient-masked
artifacts, which is the paper's point (2/4 base seeds collapse, 0/4 with the
guard). The paper reports the base row via these survivors and cross-references
the collapsed seeds rather than averaging a bimodal distribution.

**CIFAR-100, ε=8/255, ResNet18 (γ_max=1.25, m_max=4.5, λ_d=0, guard θ=0.08, probes on)**

| dir | AA | CR |
|---|---|---|
| `cifar100/guard/s0` | 27.20 | 27.30 |
| `cifar100/guard/s1` | 27.13 | 27.14 |
| `cifar100/guard/s2` | 27.13 | 27.21 |
| `cifar100/guard/s3` | 27.13 | 27.15 |
| **mean ± std** | **27.15 ± 0.04** | **27.20 ± 0.07** |

**Tiny-ImageNet, ε=8/255, PreActResNet18 (γ_max=1.0, m_max=2.5, λ_d=0, guard θ=0.08, probes off)**

| dir | AA | CR |
|---|---|---|
| `tiny-imagenet/guard/s0` | 20.49 | 20.60 |
| `tiny-imagenet/guard/s1` | 20.57 | 20.64 |
| `tiny-imagenet/guard/s2` | 20.36 | 20.37 |
| **mean ± std** | **20.47 ± 0.11** | **20.54 ± 0.15** |

CR values are printed with one decimal by the attack logger; the table above
transcribes them verbatim.

## Notes on exact reproduction

- **Script lineage.** The released runs were produced with an earlier
  version of this codebase; the first line of each `train.log` records the
  launch tag used for that run (`axis_pilot variant=dc`). The repo's
  `train.py` is that code path with the exploration variants and hardcoded
  paths removed; per-step computation and RNG order are unchanged.

- **Probe lottery.** The batch probes consume RNG, so `--probe-every 50`
  (default) vs `--probe-every 0` yields *different* trajectories for the same
  seed — instrumentation changes the CO lottery, a finding the paper reports.
  Which setting each released run used is listed in the tables above; set it
  accordingly if you want the same stream.
- **Guard-fire evidence.** `grep FIRE logs/cifar10/guard/s2/train.log` shows
  the four fire events (ep 19/22/25/98, spaced exactly one cooldown apart) and
  the recovery to the same level as the healthy seeds.
- **AA protocol.** All `autoattack_full.log` files are the complete standard
  suite on all 10 000 test images at bs=200 — the same protocol as the
  FastAT_Benchmark released numbers.
