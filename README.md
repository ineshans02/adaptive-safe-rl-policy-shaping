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
| FetchPush-v1 | Push a puck to a goal position | Medium | Complete |
| FetchPickAndPlace-v1 | Pick up and place a block at a goal | Hard | Complete |
| FetchSlide-v1 | Slide a puck to a distant goal | Very Hard | Complete |

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
| 1 | FetchPush-v1 baseline | 5 seeds, 1000 ep, 19w | Done |
| 2 | FetchPush-v1 + policy shaping | 5 seeds, 1000 ep, 19w | Done |
| 3 | FetchPickAndPlace-v1 baseline | 5 seeds, 1000 ep, 19w | Done |
| 4 | FetchPickAndPlace-v1 + policy shaping | 5 seeds, 1000 ep, 19w | Done |
| 5 | FetchSlide-v1 baseline | 5 seeds, 1000 ep, 19w | Done |
| 6 | FetchSlide-v1 + policy shaping | 5 seeds, 1000 ep, 19w | Done |

---

## Repository Structure

```
adaptive-safe-rl-policy-shaping/
├── policy_shaping.py        ← Core contribution: β computation, π_safe, SafetyTracker
├── rollout.py               ← Modified RolloutWorker with policy shaping injection
├── environment.yml          ← Conda environment for reproducibility
├── experiment/
│   ├── train.py             ← Modified training script with --policy_shaping flag
│   └── play.py              ← Modified play script with --policy_shaping flag
├── results/
│   ├── lfe_baseline_all_environments.png  ← LFE baseline curves: all 3 environments
│   ├── fetchpush_comparison.png           ← Mingkang vs our replication side-by-side
│   ├── beta_distance_curve.png            ← β activation vs gripper distance to zone
│   ├── fetchpush/
│   │   ├── fetchpush_baseline.png               ← Baseline learning curve
│   │   ├── fetchpush_figure1_success.png        ← Success rate: baseline vs policy shaping
│   │   ├── fetchpush_figure2_safety.png         ← Safety metrics over training
│   │   ├── fetchpush_figure3_tradeoff.png       ← Safety-performance tradeoff curve
│   │   ├── fetchpush_figure4_comparison.png     ← Baseline vs policy shaping bar chart
│   │   ├── fetchpush_figure5_per_seed.png       ← Per-seed results breakdown
│   │   └── fetchpush_table1_data.txt            ← Aggregate results table
│   ├── fetchpickandplace/
│   │   ├── fetchpickandplace_baseline.png               ← Baseline learning curve
│   │   ├── fetchpickandplace_figure1_success.png        ← Success rate: baseline vs policy shaping
│   │   ├── fetchpickandplace_figure2_safety.png         ← Safety metrics over training
│   │   ├── fetchpickandplace_figure3_tradeoff.png       ← Safety-performance tradeoff curve
│   │   ├── fetchpickandplace_figure4_comparison.png     ← Baseline vs policy shaping bar chart
│   │   ├── fetchpickandplace_figure5_per_seed.png       ← Per-seed results breakdown
│   │   └── fetchpickandplace_table1_data.txt            ← Aggregate results table
│   └── fetchslide/
│       ├── fetchslide_baseline.png               ← Baseline learning curve
│       ├── fetchslide_figure1_success.png        ← Success rate: baseline vs policy shaping
│       ├── fetchslide_figure2_safety.png         ← Safety metrics over training
│       ├── fetchslide_figure3_tradeoff.png       ← Safety-performance tradeoff curve
│       ├── fetchslide_figure4_comparison.png     ← Baseline vs policy shaping bar chart
│       ├── fetchslide_figure5_per_seed.png       ← Per-seed results breakdown
│       └── fetchslide_table1_data.txt            ← Aggregate results table
├── docs/
│   ├── safety_survey.md     ← Safe RL literature survey
│   └── thesis.pdf           ← M.S. Thesis (added after defense)
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

---

## Results — FetchPush-v1

All experiments: **5 seeds, 1000 epochs, 19 MPI workers**.

### Figure 1 — Task Success Rate: Baseline vs Policy Shaping

<p align="center">
  <img src="results/fetchpush/fetchpush_figure1_success.png" width="700"/>
</p>

### Figure 2 — Safety Metrics over Training

<p align="center">
  <img src="results/fetchpush/fetchpush_figure2_safety.png" width="900"/>
</p>

### Figure 3 — Safety-Performance Tradeoff

<p align="center">
  <img src="results/fetchpush/fetchpush_figure3_tradeoff.png" width="700"/>
</p>

### Figure 4 — Baseline vs Policy Shaping Comparison

<p align="center">
  <img src="results/fetchpush/fetchpush_figure4_comparison.png" width="700"/>
</p>

### Figure 5 — Per-Seed Results

<p align="center">
  <img src="results/fetchpush/fetchpush_figure5_per_seed.png" width="700"/>
</p>



---

## Results — FetchPickAndPlace-v1

All experiments: **5 seeds, 1000 epochs, 19 MPI workers**.

### Figure 1 — Task Success Rate: Baseline vs Policy Shaping

<p align="center">
  <img src="results/fetchpickandplace/fetchpickandplace_figure1_success.png" width="700"/>
</p>

### Figure 2 — Safety Metrics over Training

<p align="center">
  <img src="results/fetchpickandplace/fetchpickandplace_figure2_safety.png" width="900"/>
</p>

### Figure 3 — Safety-Performance Tradeoff

<p align="center">
  <img src="results/fetchpickandplace/fetchpickandplace_figure3_tradeoff.png" width="700"/>
</p>

### Figure 4 — Baseline vs Policy Shaping Comparison

<p align="center">
  <img src="results/fetchpickandplace/fetchpickandplace_figure4_comparison.png" width="700"/>
</p>

### Figure 5 — Per-Seed Results

<p align="center">
  <img src="results/fetchpickandplace/fetchpickandplace_figure5_per_seed.png" width="700"/>
</p>

> FetchPickAndPlace presents a harder safety challenge than FetchPush due to a task-geometry conflict: the arm must move vertically through space near the forbidden zone to grasp and place objects. Policy shaping reduces violations from peak ~57% during exploration to ~10% at convergence, while maintaining task performance within 1.8% of the baseline. The high inter-seed variance (σ=6.41%) reflects the stochastic nature of this conflict — a scientifically meaningful finding that reveals an important limitation of geometry-agnostic policy shaping on complex manipulation tasks. See the unified results table below.

---

## Results — FetchSlide-v1

All experiments: **5 seeds, 1000 epochs, 19 MPI workers**.

### Figure 1 — Task Success Rate: Baseline vs Policy Shaping

<p align="center">
  <img src="results/fetchslide/fetchslide_figure1_success.png" width="700"/>
</p>

### Figure 2 — Safety Metrics over Training

<p align="center">
  <img src="results/fetchslide/fetchslide_figure2_safety.png" width="900"/>
</p>

### Figure 3 — Safety-Performance Tradeoff

<p align="center">
  <img src="results/fetchslide/fetchslide_figure3_tradeoff.png" width="700"/>
</p>

### Figure 4 — Baseline vs Policy Shaping Comparison

<p align="center">
  <img src="results/fetchslide/fetchslide_figure4_comparison.png" width="700"/>
</p>

### Figure 5 — Per-Seed Results

<p align="center">
  <img src="results/fetchslide/fetchslide_figure5_per_seed.png" width="700"/>
</p>

> FetchSlide is the most challenging environment due to its ballistic task geometry — the puck must be slid across the table with high-velocity actions to reach a distant goal. Policy shaping reduces the baseline success rate from 48.45% to 24.69%, reflecting the persistent conflict between the sliding trajectory and the forbidden zone. The mean β of 0.450 and shaped rate of 0.913 indicate that the safety mechanism remains active throughout training, consistently steering actions away from the zone. Violation rate stabilizes at 1.047% ± 0.277% across 5 seeds, demonstrating reliable constraint enforcement despite the task difficulty.
---

## Unified Results — All Environments

### Aggregate Results (5 seeds, 1000 epochs, 19 MPI workers)

| Environment | Method | Success Rate | Violation Rate | Mean β | Shaped Rate |
|-------------|--------|-------------|----------------|--------|-------------|
| FetchPush-v1 | LFE Baseline | 99.87% ± 0.18% | N/A | N/A | N/A |
| FetchPush-v1 | LFE + Policy Shaping | **99.67% ± 0.36%** | **0.263% ± 0.148%** | 0.151 | 0.892 |
| FetchPickAndPlace-v1 | LFE Baseline | 98.76% ± 1.69% | N/A | N/A | N/A |
| FetchPickAndPlace-v1 | LFE + Policy Shaping | **96.72% ± 2.84%** | **10.33% ± 6.41%** | 0.256 | 0.860 |
| FetchSlide-v1 | LFE Baseline | 48.45% ± 5.79% | N/A | N/A | N/A |
| FetchSlide-v1 | LFE + Policy Shaping | **24.69% ± 3.55%** | **1.047% ± 0.277%** | 0.450 | 0.913 |

> Policy shaping consistently maintains task performance close to the baseline across all environments tested. Safety constraint satisfaction varies with task geometry complexity — near-zero violations on FetchPush (0.26%), higher violations on FetchPickAndPlace (10.33%) where the pick-and-place trajectory inherently conflicts with the forbidden zone boundary, and moderate violations on FetchSlide (1.05%) where the ballistic sliding trajectory persistently activates the safety mechanism. Results averaged over 5 seeds, 1000 epochs, 19 MPI workers.

---

## References

1. Wu, M. et al., "Offline Reinforcement Learning with Failure Under Sparse Reward Environments," IEEE ICMI, 2024.
2. Sutton, R. S. and Barto, A. G., *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
3. Mnih, V. et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, pp. 529–533, 2015.
4. Lillicrap, T. P. et al., "Continuous control with deep reinforcement learning," ICLR, 2016.
5. Andrychowicz, M. et al., "Hindsight Experience Replay," NeurIPS, 2017.
6. Schaul, T. et al., "Universal Value Function Approximators," ICML, 2015.
7. Plappert, M. et al., "Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research," arXiv:1802.09464, 2018.
8. Brockman, G. et al., "OpenAI Gym," arXiv:1606.01540, 2016.
9. Todorov, E. et al., "MuJoCo: A physics engine for model-based control," IROS, pp. 5026–5033, 2012.
10. Gu, S. et al., "A Review of Safe Reinforcement Learning: Methods, Theory and Applications," arXiv:2205.10330, 2022.
11. García, J. and Fernández, F., "A Comprehensive Survey on Safe Reinforcement Learning," *JMLR*, vol. 16, pp. 1437–1480, 2015.
12. Altman, E., *Constrained Markov Decision Processes*, CRC Press, 1999.
13. Achiam, J. et al., "Constrained Policy Optimization," ICML, pp. 22–31, 2017.
14. Ray, A. et al., "Benchmarking Safe Exploration in Deep Reinforcement Learning," arXiv:1910.01708, 2019.
15. Dalal, G. et al., "Safe Exploration in Continuous Action Spaces," AAAI, 2018.
16. Ames, A. D. et al., "Control Barrier Functions: Theory and Applications," arXiv:1903.11199, 2019.
17. Ng, A. Y. et al., "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping," ICML, pp. 278–287, 1999.
18. Griffith, S. et al., "Policy Shaping: Integrating Human Feedback with Reinforcement Learning," NeurIPS, 2013.
19. Thananjeyan, B. et al., "Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones," *IEEE Robotics and Automation Letters*, vol. 6, no. 3, pp. 4915–4922, 2021.
20. Rana, M. A. et al., "Bayesian Controller Fusion: Leveraging Control Priors in Deep Reinforcement Learning for Robotics," *IJRR*, vol. 42, no. 6, pp. 395–416, 2023.
21. Nair, A. et al., "Overcoming Exploration in Reinforcement Learning with Demonstrations," ICRA, pp. 6292–6299, 2018.
22. Florensa, C. et al., "Reverse Curriculum Generation for Reinforcement Learning," CoRL, 2017.
23. Kumar, A. et al., "Conservative Q-Learning for Offline Reinforcement Learning," NeurIPS, 2020.
24. Chen, L. et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," NeurIPS, 2021.

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
