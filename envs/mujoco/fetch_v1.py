import gym
import numpy as np
from gym.envs.mujoco import mujoco_env
from gym.wrappers.time_limit import TimeLimit
from envs.mujoco.mujoco_utils import MujocoTrait

class FetchEnv(MujocoTrait, mujoco_env.MujocoEnv):
    def __init__(self,
                 task="FetchPush-v1",
                 max_episode_steps=50,
                 done_allowing_step_unit=None,
                 render_mode='rgb_array',
                 ):

        self.env = gym.make(f'{task}')
        if max_episode_steps is not None:
            self.env = self.env.env
            self.env = TimeLimit(self.env, max_episode_steps=max_episode_steps)

        # print("self.env:", self.env)
        # self._action_space = self.env.action_space
        # print("action_space:", self._action_space)
        # print("env.spec: ", self.env.spec)

        # original_obs_space = self.observation_space
        # self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # print("action_space: ", self._action_space)

        # self.spec = gym.envs.registration.EnvSpec(id=f'{task}', max_episode_steps=max_episode_steps)
        # self.spec.observation_space = self._observation_space
        # self.spec.action_space = self.env.action_space

        # if hasattr(self.env, 'spec') and self.env.spec is not None:
        #     self.env.spec.observation_space = self._observation_space
        #     self.env.spec.action_space = self._action_space
        #     print("self.env.spec.action_space: ", self.env.spec.action_space)

        self._viewers = {}
        self._task = task
        self._done_allowing_step_unit = done_allowing_step_unit

    @property
    def observation_space(self):
        # ############################
        # print("observation_space: ", self._observation_space)
        # ############################
        return self.env.observation_space
    
    @property
    def action_space(self):
        # ############################
        # print("action_space: ", self.env.action_space)
        # ############################
        return self.env.action_space

    def observation(self, obs_dict):
        obs_dict['observation'] = obs_dict['observation'][:6]
        return obs_dict

    def reset(self, **kwargs):
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs_dict = reset_result[0]
        else:
            obs_dict = reset_result
        obs = self.observation(obs_dict)
        return obs


    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs_dict, reward, done, truncated, info = result
        else:
            obs_dict, reward, done, info = result
            truncated = False
        obs = self.observation(obs_dict)
        if 'TimeLimit.truncated' in info:
            del info['TimeLimit.truncated']
            
        return obs, reward, done, info

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        coord_dims = [0, 1]
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories, coord_dims)
        return eval_metrics