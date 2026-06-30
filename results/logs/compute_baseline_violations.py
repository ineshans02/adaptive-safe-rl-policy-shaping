"""
compute_baseline_violations.py - Fixed version
Runs each seed in a separate subprocess to avoid TF graph conflicts.

Usage: python compute_baseline_violations.py
"""

import os
import sys
import subprocess
import numpy as np
import json

ZONES = {
    'push':         {'x': (1.05, 1.20), 'y': (0.60, 0.90), 'z': (0.40, 0.45)},
    'pickandplace': {'x': (1.05, 1.20), 'y': (0.60, 0.90), 'z': (0.40, 0.60)},
    'slide':        {'x': (1.05, 1.20), 'y': (0.60, 0.90), 'z': (0.40, 0.45)},
}

ENV_NAMES = {
    'push':         'FetchPush-v1',
    'pickandplace': 'FetchPickAndPlace-v1',
    'slide':        'FetchSlide-v1',
}

BASELINE_DIR = os.path.expanduser('~/lfe_results/thesis_baseline')
N_EPISODES   = 100
SEEDS        = [0, 1, 2, 3, 4]

# ── Single-seed worker script written inline ──────────────────────────────────
WORKER_SCRIPT = '''
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, os.path.expanduser('~/Learning-from-Failure-for-Robotic-Arms/experiment'))
sys.path.insert(0, os.path.expanduser('~/Learning-from-Failure-for-Robotic-Arms'))

env_key    = sys.argv[1]
seed       = int(sys.argv[2])
n_episodes = int(sys.argv[3])

ZONES = {
    'push':         {'x': (1.05, 1.20), 'y': (0.60, 0.90), 'z': (0.40, 0.45)},
    'pickandplace': {'x': (1.05, 1.20), 'y': (0.60, 0.90), 'z': (0.40, 0.60)},
    'slide':        {'x': (1.05, 1.20), 'y': (0.60, 0.90), 'z': (0.40, 0.45)},
}
ENV_NAMES = {
    'push':         'FetchPush-v1',
    'pickandplace': 'FetchPickAndPlace-v1',
    'slide':        'FetchSlide-v1',
}

zone = ZONES[env_key]
policy_path = os.path.expanduser(
    f'~/lfe_results/thesis_baseline/{env_key}/seed_{seed}/policy_latest.pkl')

def in_zone(pos):
    x, y, z = pos
    return (zone["x"][0] <= x <= zone["x"][1] and
            zone["y"][0] <= y <= zone["y"][1] and
            zone["z"][0] <= z <= zone["z"][1])

with open(policy_path, "rb") as f:
    policy = pickle.load(f)

import gym
env = gym.make(ENV_NAMES[env_key])

total_ts = 0
viol_ts  = 0
successes = 0

for ep in range(n_episodes):
    obs  = env.reset()
    done = False
    while not done:
        gripper = obs["observation"][:3]
        if in_zone(gripper):
            viol_ts += 1
        total_ts += 1
        action = policy.get_actions(
            obs["observation"], obs["achieved_goal"], obs["desired_goal"])
        obs, reward, done, info = env.step(action)
        if info.get("is_success", 0):
            successes += 1
            done = True

env.close()

result = {
    "success_rate":   successes / n_episodes * 100,
    "violation_rate": viol_ts / total_ts * 100,
}
print(json.dumps(result))
'''

# Write worker script to disk
worker_path = os.path.expanduser('~/lfe_results/_worker.py')
with open(worker_path, 'w') as f:
    f.write(WORKER_SCRIPT)

# ── Main loop ─────────────────────────────────────────────────────────────────
all_results = {}

for env_key in ['push', 'pickandplace', 'slide']:
    print(f'\n{"="*60}')
    print(f'Environment: {ENV_NAMES[env_key]}')
    print(f'{"="*60}')

    seed_violations = []
    seed_successes  = []

    for seed in SEEDS:
        print(f'  Seed {seed}...', end=' ', flush=True)
        try:
            result = subprocess.run(
                [sys.executable, worker_path, env_key, str(seed), str(N_EPISODES)],
                capture_output=True, text=True, timeout=300
            )
            # Find the JSON line in stdout
            json_line = None
            for line in result.stdout.strip().split('\n'):
                try:
                    json_line = json.loads(line)
                    break
                except:
                    continue

            if json_line:
                sr = json_line['success_rate']
                vr = json_line['violation_rate']
                seed_successes.append(sr)
                seed_violations.append(vr)
                print(f'success={sr:.1f}%  violation={vr:.3f}%')
            else:
                print(f'ERROR (no JSON output)')
                print(f'    stderr: {result.stderr[-200:]}')
        except Exception as e:
            print(f'ERROR: {e}')

    if seed_violations:
        mean_v = np.mean(seed_violations)
        std_v  = np.std(seed_violations)
        mean_s = np.mean(seed_successes)
        std_s  = np.std(seed_successes)
        print(f'\n  --- {ENV_NAMES[env_key]} Baseline Summary ---')
        print(f'  Success Rate:   {mean_s:.2f}% ± {std_s:.2f}%')
        print(f'  Violation Rate: {mean_v:.3f}% ± {std_v:.3f}%')
        all_results[env_key] = {
            'mean_success': mean_s, 'std_success': std_s,
            'mean_violation': mean_v, 'std_violation': std_v,
            'per_seed_success': seed_successes,
            'per_seed_violation': seed_violations,
        }

# ── Final Summary ─────────────────────────────────────────────────────────────
print(f'\n\n{"="*70}')
print('BASELINE VIOLATION RATES — All Environments')
print(f'{"="*70}')
print(f'{"Environment":<25} {"Success Rate":>14} {"Violation Rate":>16}')
print(f'{"-"*70}')
for env_key, r in all_results.items():
    print(f'{ENV_NAMES[env_key]:<25} '
          f'{r["mean_success"]:>11.2f}%±{r["std_success"]:.2f}% '
          f'{r["mean_violation"]:>13.3f}%±{r["std_violation"]:.3f}%')
print(f'{"="*70}')

# Save results to file
out_path = os.path.expanduser('~/lfe_results/baseline_violation_results.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'\nResults saved to: {out_path}')
