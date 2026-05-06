from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from util import convert_episode_to_batch_major, store_args

# ── Policy Shaping Import ──────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy_shaping import shape_action, SafetyTracker, ZONES


class RolloutWorker:

    @store_args
    def __init__(self, madeEnv, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False,
                 policy_shaping=False, env_name='FetchPush-v1',
                 d_threshold=0.15, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            madeEnv         : for gazebo envs multiple envs are not possible yet
            make_env        : factory function that creates a new environment instance
            policy          : the policy that is used to act
            dims            : dict of dimensions for observations, goals, actions
            logger          : logger used by the rollout worker
            T               : time horizon
            rollout_batch_size : number of parallel rollouts
            exploit         : whether to act optimally (no exploration)
            use_target_net  : whether to use target net for rollouts
            compute_Q       : whether to compute Q values alongside actions
            noise_eps       : scale of additive Gaussian noise
            random_eps      : probability of selecting a completely random action
            history_len     : length of history for statistics smoothing
            render          : whether to render rollouts
            policy_shaping  : whether to enable policy shaping safety mechanism
            env_name        : environment name for zone config lookup
            d_threshold     : safety activation distance in metres
        """
        self.envs = [madeEnv]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)
        self.reset_all_rollouts()
        self.clear_history()

        # ── Safety tracker (one per worker) ──
        self.safety_tracker = SafetyTracker()
        if self.policy_shaping:
            print(f"[PolicyShaping] ENABLED for {self.env_name}, d_threshold={self.d_threshold}m")
            zone = ZONES.get(self.env_name, ZONES['FetchPush-v1'])
            print(f"[PolicyShaping] Forbidden zone: x=[{zone['x_min']},{zone['x_max']}], "
                  f"y=[{zone['y_min']},{zone['y_max']}], "
                  f"z=[{zone['z_min']},{zone['z_max']}]")

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment."""
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers."""
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T`."""
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []

        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)

            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # ── Policy Shaping Injection ───────────────────────────────
                    # Apply shaped action if policy_shaping is enabled.
                    # u[i] is the raw LFE task policy action (4-dim vector).
                    # shape_action blends it with pi_safe based on beta(s).
                    # The shaped action is what actually gets executed.
                    if self.policy_shaping:
                        u_shaped, safety_info = shape_action(
                            u[i].copy(),
                            o[i],                  # current observation (25-dim)
                            self.env_name,
                            d_threshold=self.d_threshold,
                            enabled=True
                        )
                        self.safety_tracker.record_step(safety_info)
                        action_to_execute = u_shaped
                    else:
                        action_to_execute = u[i]
                    # ── End Policy Shaping ─────────────────────────────────────

                    curr_o_new, _, _, info = self.envs[i].step(action_to_execute)

                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())      # NOTE: store original u, not shaped action
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        # ── End of episode: record safety stats ───────────────────────────────
        if self.policy_shaping:
            self.safety_tracker.end_episode()

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics."""
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection."""
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics."""
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        # ── Safety logs ───────────────────────────────────────────────────────
        if self.policy_shaping and len(self.safety_tracker.episode_violations) > 0:
            logs += [('safety/violation_rate',
                      np.mean(self.safety_tracker.episode_violations[-100:]))]
            logs += [('safety/mean_beta',
                      np.mean(self.safety_tracker.episode_beta_means[-100:]))]
            logs += [('safety/shaped_rate',
                      np.mean(self.safety_tracker.episode_shaped_rate[-100:]))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed."""
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)


class RolloutWorkerOriginal:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False,
                 policy_shaping=False, env_name='FetchPush-v1',
                 d_threshold=0.15, **kwargs):
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)
        self.reset_all_rollouts()
        self.clear_history()

        # ── Safety tracker ────────────────────────────────────────────────────
        self.safety_tracker = SafetyTracker()
        if self.policy_shaping:
            print(f"[PolicyShaping] ENABLED for {self.env_name}, d_threshold={self.d_threshold}m")
            zone = ZONES.get(self.env_name, ZONES['FetchPush-v1'])
            print(f"[PolicyShaping] Forbidden zone: x=[{zone['x_min']},{zone['x_max']}], "
                  f"y=[{zone['y_min']},{zone['y_max']}], "
                  f"z=[{zone['z_min']},{zone['z_max']}]")

    def reset_rollout(self, i):
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']
        self.obs_0 = obs

    def reset_all_rollouts(self):
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        self.reset_all_rollouts()

        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        obs_d, achieved_goals_d, acts_d, goals_d, successes_d = [], [], [], [], []
        obs_dd, acts_dd, successes_dd = [], [], []
        obs_d.append(self.obs_0)
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []

        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                u = u.reshape(1, -1)

            obsnew, rewardnew, donenew, infonew = self.envs[0].step(u.tolist()[0])
            obs_d.append(obsnew)
            acts_d.append(u.tolist()[0])
            successes_d.append(infonew)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)

            for i in range(self.rollout_batch_size):
                try:
                    # ── Policy Shaping Injection ───────────────────────────────
                    if self.policy_shaping:
                        u_shaped, safety_info = shape_action(
                            u[i].copy(),
                            o[i],
                            self.env_name,
                            d_threshold=self.d_threshold,
                            enabled=True
                        )
                        self.safety_tracker.record_step(safety_info)
                        action_to_execute = u_shaped
                    else:
                        action_to_execute = u[i]
                    # ── End Policy Shaping ─────────────────────────────────────

                    curr_o_new, _, _, info = self.envs[i].step(action_to_execute)
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())      # store original u not shaped action
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new

        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        # ── End of episode: record safety stats ───────────────────────────────
        if self.policy_shaping:
            self.safety_tracker.end_episode()

        obs_dd.append(obs_d)
        acts_dd.append(acts_d)
        successes_dd.append(successes_d)

        print("---------------------------------------------------")
        print("---------------------------------------------------")
        actions = acts_dd
        print("acts:", type(actions), len(actions), type(actions[0]), len(actions[0]),
              len(actions[0][0]), type(actions[0][0]), type(actions[0][0][0]))
        print("---------------------------------------------------")

        fileName = "data_push"
        fileName += ".npz"
        np.savez_compressed(fileName, acs=acts_dd, obs=obs_dd, info=successes_dd)

        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        # ── Safety logs ───────────────────────────────────────────────────────
        if self.policy_shaping and len(self.safety_tracker.episode_violations) > 0:
            logs += [('safety/violation_rate',
                      np.mean(self.safety_tracker.episode_violations[-100:]))]
            logs += [('safety/mean_beta',
                      np.mean(self.safety_tracker.episode_beta_means[-100:]))]
            logs += [('safety/shaped_rate',
                      np.mean(self.safety_tracker.episode_shaped_rate[-100:]))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)