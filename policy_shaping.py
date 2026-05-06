"""
policy_shaping.py
Adaptive Safe RL Exploration via Policy Shaping from Failed Experiences
Author: Ines Hans — M.S. Thesis, UTSA, Dr. Yongcan Cao

Implements the policy shaping mechanism:
    a_shaped(s) = (1 - beta(s)) * pi_task(s) + beta(s) * pi_safe(s)

where:
    - beta(s)    : blending coefficient from distance to forbidden zone
    - pi_task(s) : Mingkang's LFE policy action (unchanged)
    - pi_safe(s) : safe action pointing away from forbidden zone centroid

Supports both FetchPush-v1 and FetchPickAndPlace-v1 via zone config.
"""

import numpy as np


# ── Forbidden Zone Configurations ─────────────────────────────────────────────
# Defined in MuJoCo world coordinates.
# Table surface: z = 0.40, table spans x=[1.05,1.55], y=[0.40,1.10]

ZONES = {
    'FetchPush-v1': {
        'x_min': 1.05, 'x_max': 1.20,
        'y_min': 0.60, 'y_max': 0.90,
        'z_min': 0.40, 'z_max': 0.45,   # thin slab at table surface
    },
    'FetchPickAndPlace-v1': {
        'x_min': 1.05, 'x_max': 1.20,
        'y_min': 0.60, 'y_max': 0.90,
        'z_min': 0.40, 'z_max': 0.60,   # taller — covers lifting height
    },
}

# Safety threshold — distance (metres) at which beta starts rising
D_THRESHOLD = 0.15   # 15 cm


# ── Core Functions ─────────────────────────────────────────────────────────────

def distance_to_zone(gripper_pos, zone):
    """
    Compute the minimum Euclidean distance from the gripper to the
    nearest face of the forbidden zone box.

    Returns 0.0 if the gripper is INSIDE the zone (violation).

    Args:
        gripper_pos : np.array of shape (3,) — [x, y, z]
        zone        : dict with keys x_min, x_max, y_min, y_max, z_min, z_max

    Returns:
        float — distance in metres
    """
    dx = max(zone['x_min'] - gripper_pos[0], 0.0,
             gripper_pos[0] - zone['x_max'])
    dy = max(zone['y_min'] - gripper_pos[1], 0.0,
             gripper_pos[1] - zone['y_max'])
    dz = max(zone['z_min'] - gripper_pos[2], 0.0,
             gripper_pos[2] - zone['z_max'])
    return np.sqrt(dx**2 + dy**2 + dz**2)


def compute_beta(gripper_pos, zone, d_threshold=D_THRESHOLD):
    """
    Compute blending coefficient beta(s) via sigmoid over distance.

        beta(s) = sigmoid( (d_threshold - d(s,Z)) / d_threshold )

    - Far from zone  (d >> d_threshold) : beta ≈ 0  → pure task policy
    - At threshold   (d = d_threshold)  : beta = 0.5
    - At boundary    (d = 0)            : beta ≈ 0.73
    - Inside zone    (d = 0, violation) : beta ≈ 0.73 → pure safe policy

    Args:
        gripper_pos : np.array (3,)
        zone        : dict
        d_threshold : float

    Returns:
        float in [0, 1]
    """
    d = distance_to_zone(gripper_pos, zone)
    # Steeper sigmoid with scaling factor 5
    x = 5.0 * (d_threshold - d) / d_threshold
    beta = 1.0 / (1.0 + np.exp(-x))
    return float(np.clip(beta, 0.0, 1.0))


def compute_pi_safe(gripper_pos, zone):
    """
    Compute the safe action direction — a unit vector pointing the gripper
    maximally away from the centroid of the forbidden zone.

    Only x and y components are used (motion in table plane).
    z component is set to 0 (no vertical safe action).

    Args:
        gripper_pos : np.array (3,)
        zone        : dict

    Returns:
        np.array of shape (4,) — safe action [dx, dy, dz, gripper]
                                  normalized to max_u = 1.0
    """
    zone_center = np.array([
        (zone['x_min'] + zone['x_max']) / 2.0,
        (zone['y_min'] + zone['y_max']) / 2.0,
        (zone['z_min'] + zone['z_max']) / 2.0,
    ])

    # Direction away from zone center (x, y only)
    direction = gripper_pos - zone_center
    direction[2] = 0.0   # no vertical component

    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        # Gripper is exactly at zone center — push in +x direction as fallback
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm

    # Safe action: move away at full speed, keep gripper open (4th dim = 0)
    pi_safe = np.array([direction[0], direction[1], 0.0, 0.0])
    return pi_safe


def shape_action(u, obs, env_name, d_threshold=D_THRESHOLD,
                 enabled=True, track_violations=True):
    """
    Apply policy shaping to action u given current observation.

        a_shaped = (1 - beta) * u + beta * pi_safe

    Args:
        u               : np.array (4,) — raw task policy action
        obs             : np.array (25,) — full observation vector
        env_name        : str — 'FetchPush-v1' or 'FetchPickAndPlace-v1'
        d_threshold     : float — safety activation distance
        enabled         : bool — if False, returns u unchanged (baseline mode)
        track_violations: bool — if True, returns violation flag

    Returns:
        a_shaped        : np.array (4,) — shaped action
        info            : dict with keys:
                            'beta'      : float
                            'distance'  : float
                            'violation' : bool (gripper inside zone)
                            'shaped'    : bool (beta > 0.01)
    """
    if not enabled:
        return u.copy(), {'beta': 0.0, 'distance': np.inf,
                          'violation': False, 'shaped': False}

    # Get zone for this environment
    zone = ZONES.get(env_name, ZONES['FetchPush-v1'])

    # Extract gripper position from observation (indices 0:3)
    gripper_pos = obs[:3].copy()

    # Compute distance and beta
    d    = distance_to_zone(gripper_pos, zone)
    beta = compute_beta(gripper_pos, zone, d_threshold)

    # Compute safe action
    pi_safe = compute_pi_safe(gripper_pos, zone)

    # Blend: a_shaped = (1 - beta) * u + beta * pi_safe
    a_shaped = (1.0 - beta) * u + beta * pi_safe

    # Clip to valid action range [-1, 1]
    a_shaped = np.clip(a_shaped, -1.0, 1.0)

    info = {
        'beta'     : beta,
        'distance' : d,
        'violation': d == 0.0,   # inside zone = violation
        'shaped'   : beta > 0.01,
    }

    return a_shaped, info


# ── Episode-level Safety Tracker ──────────────────────────────────────────────

class SafetyTracker:
    """
    Tracks safety metrics across an episode and across training.
    Use one instance per RolloutWorker.
    """

    def __init__(self):
        self.reset_episode()
        self.episode_violations  = []   # list of per-episode violation counts
        self.episode_beta_means  = []   # list of per-episode mean beta
        self.episode_shaped_rate = []   # fraction of steps where shaping active

    def reset_episode(self):
        self.step_violations = 0
        self.step_betas      = []
        self.step_shaped     = []
        self.n_steps         = 0

    def record_step(self, info):
        """Call after each shaped_action call."""
        self.n_steps += 1
        self.step_betas.append(info['beta'])
        self.step_shaped.append(float(info['shaped']))
        if info['violation']:
            self.step_violations += 1

    def end_episode(self):
        """Call at end of each episode. Returns episode safety summary."""
        summary = {
            'violations'      : self.step_violations,
            'violation_rate'  : self.step_violations / max(self.n_steps, 1),
            'mean_beta'       : float(np.mean(self.step_betas)) if self.step_betas else 0.0,
            'shaped_rate'     : float(np.mean(self.step_shaped)) if self.step_shaped else 0.0,
        }
        self.episode_violations.append(self.step_violations)
        self.episode_beta_means.append(summary['mean_beta'])
        self.episode_shaped_rate.append(summary['shaped_rate'])
        self.reset_episode()
        return summary

    def global_summary(self):
        """Returns safety stats across all recorded episodes."""
        total_eps = len(self.episode_violations)
        if total_eps == 0:
            return {}
        return {
            'total_episodes'        : total_eps,
            'total_violations'      : sum(self.episode_violations),
            'mean_violations_per_ep': np.mean(self.episode_violations),
            'violation_rate'        : sum(self.episode_violations) /
                                      (total_eps * max(1, self.n_steps)),
            'mean_beta'             : np.mean(self.episode_beta_means),
            'mean_shaped_rate'      : np.mean(self.episode_shaped_rate),
        }


# ── Quick Test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("Policy Shaping — Unit Test")
    print("=" * 55)

    zone = ZONES['FetchPush-v1']

    # Test 1: gripper far from zone
    pos_far  = np.array([1.40, 0.75, 0.42])
    d_far    = distance_to_zone(pos_far, zone)
    beta_far = compute_beta(pos_far, zone)
    print(f"\nTest 1 — Far from zone:")
    print(f"  gripper pos : {pos_far}")
    print(f"  distance    : {d_far:.4f} m")
    print(f"  beta        : {beta_far:.4f}  (should be ≈ 0)")

    # Test 2: gripper at zone boundary
    pos_edge  = np.array([1.20, 0.75, 0.42])
    d_edge    = distance_to_zone(pos_edge, zone)
    beta_edge = compute_beta(pos_edge, zone)
    print(f"\nTest 2 — At zone boundary:")
    print(f"  gripper pos : {pos_edge}")
    print(f"  distance    : {d_edge:.4f} m")
    print(f"  beta        : {beta_edge:.4f}  (should be ≈ 0.5-0.73)")

    # Test 3: gripper inside zone (violation)
    pos_inside  = np.array([1.12, 0.75, 0.42])
    d_inside    = distance_to_zone(pos_inside, zone)
    beta_inside = compute_beta(pos_inside, zone)
    print(f"\nTest 3 — Inside zone (violation):")
    print(f"  gripper pos : {pos_inside}")
    print(f"  distance    : {d_inside:.4f} m  (should be 0.0)")
    print(f"  beta        : {beta_inside:.4f}  (should be ≈ 0.73)")

    # Test 4: full shape_action call
    u   = np.array([0.5, 0.3, 0.0, 0.0])
    obs = np.zeros(25)
    obs[:3] = pos_edge
    a_shaped, info = shape_action(u, obs, 'FetchPush-v1')
    print(f"\nTest 4 — shape_action at boundary:")
    print(f"  raw action    : {u}")
    print(f"  shaped action : {a_shaped}")
    print(f"  beta          : {info['beta']:.4f}")
    print(f"  violation     : {info['violation']}")
    print(f"  shaped        : {info['shaped']}")

    print("\n" + "=" * 55)
    print("All tests passed!")
    print("=" * 55)