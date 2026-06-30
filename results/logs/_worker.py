
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
