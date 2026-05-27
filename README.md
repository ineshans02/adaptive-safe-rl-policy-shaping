# Adaptive Safe RL Exploration via Policy Shaping from Failed Experiences

**M.S. Thesis — Electrical Engineering**  
**University of Texas at San Antonio (UTSA)**  
**Author:** Ines Hans  
**Advisor:** Dr. Yongcan Cao  

---

## Overview

This repository contains the implementation of my M.S. thesis, which extends Mingkang Wu's **Learning from Failed Experiences (LFE)** framework (IEEE ICMI 2024) with a safe reinforcement learning mechanism via **offline policy shaping**.

The core contribution is a dual-purpose role for the failed experience buffer `Rf`:
1. **During training** — serves as a repulsion signal (per LFE design)
2. **At execution time** — provides a safety prior via offline policy shaping

The shaped action equation is:

$$a_{shaped}(s) = (1 - \beta(s)) \cdot \pi_{task}(s|\theta_\pi) + \beta(s) \cdot \pi_{safe}(s)$$

where:
- $\beta(s)$ is derived from the distance of the gripper to a forbidden zone via a sigmoid
- $\pi_{safe}(s)$ points maximally away from the forbidden zone centroid
- $\pi_{task}(s)$ is the LFE task policy (unchanged from Mingkang's work)

---

## Environments

Experiments are conducted on the **OpenAI Gym Fetch robotics suite** using MuJoCo:

| Environment | Task | Difficulty | Status |
|-------------|------|------------|--------|
| FetchPush-v1 | Push a puck to a goal position | Medium | Primary — Complete |
| FetchPickAndPlace-v1 | Pick up and place a block at a goal | Hard | In Progress |
| FetchSlide-v1 | Slide a puck to a distant goal | Very Hard | Pending |

---

## Safety Definition

Safety is defined as a **3D forbidden rectangular zone** on the table surface that the gripper end-effector must not enter.

| Environment | Zone (x) | Zone (y) | Zone (z) |
|-------------|----------|----------|----------|
| FetchPush-v1 | [1.05, 1.20] | [0.60, 0.90] | [0.40, 0.45] |
| FetchPickAndPlace-v1 | [1.05, 1.20] | [0.60, 0.90] | [0.40, 0.60] |
| FetchSlide-v1 | [1.05, 1.20] | [0.60, 0.90] | [0.40, 0.45] |

The blending coefficient is computed as:

$$\beta(s) = \sigma\left(\frac{5 \cdot (d_{threshold} - d(s, \mathcal{Z}))}{d_{threshold}}\right)$$

where $d(s, \mathcal{Z})$ is the Euclidean distance from the gripper to the nearest face of the forbidden zone and $d_{threshold} = 0.15m$.

---

## Safety Metrics

| Metric | Definition |
|--------|-----------|
| Violation rate | % of timesteps where gripper enters forbidden zone |
| β activation rate | % of timesteps where β > 0.01 |
| Shaped action rate | % of actions meaningfully modified by shaping |
| Success rate | % of episodes reaching the task goal |

---

## Experiment Status

| Phase | Environment | Config | Status |
|-------|-------------|--------|--------|
| 1 | FetchPush-v1 baseline | 5 seeds, 1000 ep, 19w |  Done |
| 2 | FetchPush-v1 + policy shaping | 5 seeds, 1000 ep, 19w |  Done |
| 3 | FetchPickAndPlace-v1 baseline | 5 seeds, 1000 ep, 19w |  Done |
| 4 | FetchPickAndPlace-v1 + policy shaping | 5 seeds, 1000 ep, 19w |  Running |
| 5 | FetchSlide-v1 baseline | 5 seeds, 1000 ep, 19w |  Pending |
| 6 | FetchSlide-v1 + policy shaping | 5 seeds, 1000 ep, 19w |  Pending |

---

## Repository Structure

```
adaptive-safe-rl-policy-shaping/
├── policy_shaping.py        ← Core contribution: β computation, π_safe, SafetyTracker
├── rollout.py               ← Modified RolloutWorker with policy shaping injection
├── experiment/
│   ├── train.py             ← Modified training script with --policy_shaping flag
│   └── play.py              ← Modified play script with --policy_shaping flag
├── results/
│   ├── fetchpush_comparison.png         ← Mingkang vs our replication side-by-side
│   ├── fetchpush_baseline.gif           ← FetchPush LFE baseline visualization
│   ├── fetchpush_policy_shaping.gif     ← FetchPush policy shaping with red forbidden zone
│   ├── fetchpush_figure1_success.png    ← Success rate: baseline vs policy shaping
│   ├── fetchpush_figure2_safety.png     ← Safety metrics over training
│   ├── fetchpush_figure3_tradeoff.png   ← Safety-performance tradeoff curve
│   ├── thesis_baseline_push.png         ← FetchPush 5-seed baseline curve
│   ├── plot_lfe_single.py               ← Plotting script (single graph)
│   └── plot_lfe_baseline.py             ← Plotting script (multi-subplot)
├── docs/
│   └── safety_survey.md                 ← Safe RL literature survey
└── README.md
```

---

## Key Files

### `policy_shaping.py`
The main contribution. Contains:
- `distance_to_zone()` — 3D distance from gripper to forbidden box
- `compute_beta()` — blending coefficient via sigmoid over distance
- `compute_pi_safe()` — safe action pointing away from zone centroid
- `shape_action()` — applies the full policy shaping equation
- `SafetyTracker` — tracks safety metrics across episodes and training

### `rollout.py`
Modified version of Mingkang's rollout worker. Policy shaping is injected at execution time — right after the task policy computes action `u`, before `env.step()`. The original LFE training loss is **completely unchanged**.

### `experiment/train.py`
Added two new CLI flags:
- `--policy_shaping` — enables the safety mechanism (flag, default off)
- `--d_threshold` — safety activation distance in metres (default 0.15)

---

## Installation

```bash
# Clone this repo
git clone https://github.com/ineshans02/adaptive-safe-rl-policy-shaping.git
cd adaptive-safe-rl-policy-shaping

# Clone Mingkang's LFE repo (base codebase)
git clone https://github.com/MingkangWu/Learning-from-Failure-for-Robotic-Arms.git

# Copy thesis files into LFE repo
cp policy_shaping.py Learning-from-Failure-for-Robotic-Arms/
cp rollout.py Learning-from-Failure-for-Robotic-Arms/
cp experiment/train.py Learning-from-Failure-for-Robotic-Arms/experiment/
cp experiment/play.py Learning-from-Failure-for-Robotic-Arms/experiment/

# Set up conda environment (Python 3.7, TF 1.15, MuJoCo 2.1)
conda create -n lfe python=3.7
conda activate lfe
pip install tensorflow==1.15.0
pip install gym==0.21.0
pip install mujoco-py
```

---

## Running Experiments

### LFE Baseline (no policy shaping)
```bash
cd Learning-from-Failure-for-Robotic-Arms/experiment
for seed in 0 1 2 3 4; do
  python train.py \
    --env FetchPush-v1 \
    --n_epochs 1000 \
    --num_cpu 19 \
    --seed $seed \
    --demo_file ../data_generation/data_fetch_random_100_randombad.npz \
    --logdir ~/results/lfe_baseline/push/seed_${seed} \
    2>&1 | tee ~/results/lfe_baseline/push/seed_${seed}.log
done
```

### LFE + Policy Shaping (thesis contribution)
```bash
cd Learning-from-Failure-for-Robotic-Arms/experiment
for seed in 0 1 2 3 4; do
  python train.py \
    --env FetchPush-v1 \
    --n_epochs 1000 \
    --num_cpu 19 \
    --seed $seed \
    --demo_file ../data_generation/data_fetch_random_100_randombad.npz \
    --logdir ~/results/lfe_policy_shaping/push/seed_${seed} \
    --policy_shaping \
    --d_threshold 0.15 \
    2>&1 | tee ~/results/lfe_policy_shaping/push/seed_${seed}.log
done
```

### Visualize trained policy
```bash
# Baseline
python play.py ~/results/lfe_baseline/push/seed_0/policy_best.pkl

# Policy shaping (with safety active)
python play.py ~/results/lfe_policy_shaping/push/seed_0/policy_best.pkl --policy_shaping
```

---

## Results — FetchPush-v1

All experiments: **5 seeds, 1000 epochs, 19 MPI workers**.

### Baseline replication vs Mingkang et al.

<p align="center">
  <img src="results/fetchpush_comparison.png" width="800"/>
</p>

> Left: reproduced from Wu et al. Figure 1 (5 seeds, mean ± std). Right: our replication (5 seeds, 1000 epochs, 19 MPI workers). Our replication matches Mingkang's final performance and converges significantly faster (~21 epochs vs ~200–300 epochs), confirming correct implementation before adding the policy shaping contribution.

---

### FetchPush-v1 — LFE Baseline Trained Policy (~100% success rate)

<p align="center">
  <img src="results/fetchpush_baseline.gif" width="480"/>
</p>

> *"Trained LFE policy on FetchPush-v1 — 5 seeds, 1000 epochs, 19 MPI workers. The robot arm consistently pushes the puck to the target goal position, converging to ~100% success rate in ~21 epochs."*

---

### FetchPush-v1 — Policy Shaping Trained Policy (with forbidden zone)

<p align="center">
  <img src="results/fetchpush_policy_shaping.gif" width="480"/>
</p>

> *"LFE + Policy Shaping on FetchPush-v1. The red zone represents the forbidden region the gripper must not enter. The shaped action β(s) blends the task policy with a safe policy based on proximity to the zone, achieving 99.67% ± 0.36% success rate with 0.263% ± 0.148% violation rate across 5 seeds."*

---

### Figure 1 — Task Success Rate: Baseline vs Policy Shaping

<p align="center">
  <img src="results/fetchpush_figure1_success.png" width="700"/>
</p>

### Figure 2 — Safety Metrics over Training

<p align="center">
  <img src="results/fetchpush_figure2_safety.png" width="900"/>
</p>

### Figure 3 — Safety-Performance Tradeoff

<p align="center">
  <img src="results/fetchpush_figure3_tradeoff.png" width="700"/>
</p>

### Table 1 — FetchPush-v1 Aggregate Results (last 100 epochs, 5 seeds)

| Method | Success Rate | Violation Rate | Mean β | Shaped Rate |
|--------|-------------|----------------|--------|-------------|
| LFE Baseline | 99.87% ± 0.18% | N/A | N/A | N/A |
| LFE + Policy Shaping | **99.67% ± 0.36%** | **0.263% ± 0.148%** | 0.151 | 0.892 |

> Policy shaping reduces safety violations to near zero while maintaining task performance within 0.20% of the baseline — demonstrating that safety and performance are not fundamentally at odds in this setting.

---

## Results — FetchPickAndPlace-v1

All experiments: **5 seeds, 1000 epochs, 19 MPI workers**.

### FetchPickAndPlace-v1 — LFE Baseline Learning Curve

<p align="center">
  <img src="results/fetchpickandplace_baseline.png" width="700"/>
</p>

| Method | Success Rate | Convergence Epoch | Seeds |
|--------|-------------|-------------------|-------|
| LFE Baseline | **98.76% ± 1.69%** | **~127** | 5 |
| LFE + Policy Shaping | 🔄 Training | TBD | 5 |

---

## Results — FetchSlide-v1 *(pending)*

Training pending. Results will be added upon completion.

---

## References

1. Wu, M. et al., "Offline Reinforcement Learning with Failure Under Sparse Reward Environments," IEEE ICMI 2024.
2. Nair, A. et al., "Overcoming Exploration in Reinforcement Learning with Demonstrations," ICRA 2018.
3. Andrychowicz, M. et al., "Hindsight Experience Replay," NeurIPS 2017.
4. Plappert, M. et al., "Multi-goal Reinforcement Learning," arXiv 2018.
5. Rana, K. et al., "Bayesian Controller Fusion," IJRR 2023.
6. Thananjeyan, B. et al., "Recovery RL," RAL/ICRA 2021.
7. Dalal, G. et al., "Safe Exploration in Continuous Action Spaces," 2018.

---

## Citation

```bibtex
@mastersthesis{hans2026safeLFE,
  author    = {Ines Hans},
  title     = {Adaptive Safe RL Exploration via Policy Shaping from Failed Experiences},
  school    = {University of Texas at San Antonio},
  year      = {2026},
  advisor   = {Yongcan Cao}
}
```

---

## Acknowledgments

This work builds on Mingkang Wu's LFE framework. The base codebase is available at [github.com/MingkangWu/Learning-from-Failure-for-Robotic-Arms](https://github.com/MingkangWu/Learning-from-Failure-for-Robotic-Arms).

This work was supported in part by the Army Research Lab, Army Research Office, and Office of Naval Research through Dr. Yongcan Cao's research grants.
