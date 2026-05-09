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
- $\beta(s)$ is derived from the distance of the gripper to a forbidden zone via sigmoid
- $\pi_{safe}(s)$ points maximally away from the forbidden zone centroid
- $\pi_{task}(s)$ is the LFE task policy (unchanged from Mingkang's work)

---

## Environments

Experiments are conducted on the **OpenAI Gym Fetch robotics suite** using MuJoCo:

| Environment | Task | Difficulty |
|-------------|------|------------|
| FetchPush-v1 | Push a puck to a goal position | Medium — **Primary** |
| FetchPickAndPlace-v1 | Pick and place a block | Hard — Secondary |

---

## Safety Definition

Safety is defined as a **3D forbidden rectangular zone** on the table surface that the gripper end-effector must not enter.

| Environment | Zone (x) | Zone (y) | Zone (z) |
|-------------|----------|----------|----------|
| FetchPush-v1 | [1.05, 1.20] | [0.60, 0.90] | [0.40, 0.45] |
| FetchPickAndPlace-v1 | [1.05, 1.20] | [0.60, 0.90] | [0.40, 0.60] |

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

## Repository Structure

```
adaptive-safe-rl-policy-shaping/
├── policy_shaping.py        ← Core contribution: β computation, π_safe, SafetyTracker
├── rollout.py               ← Modified RolloutWorker with policy shaping injection
├── experiment/
│   └── train.py             ← Modified training script with --policy_shaping flag
├── results/
│   ├── thesis_baseline_push.png     ← FetchPush LFE baseline (5 seeds, 1000 epochs)
│   ├── fetchpush_comparison.png     ← Side-by-side: Mingkang vs our replication
│   ├── plot_lfe_single.py           ← Plotting script (single graph)
│   └── plot_lfe_baseline.py         ← Plotting script (multi-subplot)
├── docs/
│   └── safety_survey.md             ← Safe RL literature survey
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
python train.py \
  --env FetchPush-v1 \
  --n_epochs 1000 \
  --num_cpu 19 \
  --seed 0 \
  --demo_file ../data_generation/data_fetch_random_100_randombad.npz \
  --logdir ~/results/lfe_baseline/push/seed_0
```

### LFE + Policy Shaping (thesis contribution)
```bash
cd Learning-from-Failure-for-Robotic-Arms/experiment
python train.py \
  --env FetchPush-v1 \
  --n_epochs 1000 \
  --num_cpu 19 \
  --seed 0 \
  --demo_file ../data_generation/data_fetch_random_100_randombad.npz \
  --logdir ~/results/lfe_policy_shaping/push/seed_0 \
  --policy_shaping \
  --d_threshold 0.15
```

### Multi-seed experiment (5 seeds sequentially)
```bash
for seed in 0 1 2 3 4; do
  python train.py \
    --env FetchPush-v1 \
    --n_epochs 1000 \
    --num_cpu 19 \
    --seed $seed \
    --demo_file ../data_generation/data_fetch_random_100_randombad.npz \
    --logdir ~/results/lfe_policy_shaping/push/seed_${seed} \
    --policy_shaping
done
```

### Unit test for policy shaping
```bash
cd Learning-from-Failure-for-Robotic-Arms
python policy_shaping.py
```

---

## Baseline Results — FetchPush-v1

All experiments use identical conditions: **5 seeds, 1000 epochs, 19 MPI workers**.

| Method | Success Rate | Convergence Epoch | Seeds |
|--------|-------------|-------------------|-------|
| LFE — Mingkang et al. (ICMI 2024) | ~99% ± 1% @ epoch 1000 | ~200–300 | 5 |
| LFE — Our replication | **99.87% ± 0.18%** | **~21** | 5 |
| LFE + Policy Shaping (ours) | TBD (training) | TBD | 5 |

### Side-by-side comparison

<p align="center">
  <img src="results/fetchpush_comparison.png" width="800"/>
</p>

> Left: reproduced from Wu et al. Figure 1 (5 seeds, mean ± std). Right: our replication (5 seeds, 1000 epochs, 19 MPI workers). Our replication matches Mingkang's final performance and converges significantly faster (~21 epochs vs ~200–300 epochs), confirming correct implementation before adding the policy shaping contribution.

### FetchPush-v1 — Trained Policy Visualization (LFE Baseline, 100% success rate)

<p align="center">
  <img src="results/fetchpush_baseline.gif" width="480"/>
</p>

> Trained LFE policy on FetchPush-v1 — 5 seeds, 1000 epochs, 19 MPI workers. The robot arm consistently pushes the puck to the target goal position, converging to 99.87% success rate in ~21 epochs.

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
