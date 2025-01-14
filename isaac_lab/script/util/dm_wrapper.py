from typing import Any, Dict, Optional

from dm_env import specs

import dm_env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tree
import torch
import tensorflow as tf


class GymWrapper(dm_env.Environment):
    def __init__(self, environment: gym.Env):
        self._environment = environment
        self._reset_next_step = True
        self._last_info = None

        # Convert action and observation specs.
        obs_space = self._environment.observation_space["policy"]
        act_space = self._environment.action_space
        self._observation_spec = _convert_to_spec(obs_space, name="observation")
        self._action_spec = _convert_to_spec(act_space, name="action")

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation = self._environment.reset()[0]
        observation["policy"] = squeeze_all_zero(observation["policy"])
        observation = convert_to_np(observation["policy"])
        # Reset the diagnostic information.
        self._last_info = None
        return dm_env.restart(observation)

    def step(self, action) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        num_envs = 1
        action = torch.tensor(action, dtype=torch.float, device="cuda:0").repeat(num_envs, 1)

        observation, reward, done, truncated, info = self._environment.step(action)

        observation["policy"] = squeeze_all_zero(observation["policy"])
        observation = convert_to_np(observation["policy"])
        reward = reward.cpu().numpy().item()
        done = done.cpu().numpy()
        truncated = truncated.cpu().numpy()
        info = convert_to_np(info)

        self._reset_next_step = done
        self._last_info = info

        # Convert the type of the reward based on the spec, respecting the scalar or
        # array property.
        reward = tree.map_structure(
            lambda x, t: (  # pylint: disable=g-long-lambda
                t.dtype.type(x) if np.isscalar(x) else np.asarray(x, dtype=t.dtype)
            ),
            reward,
            self.reward_spec(),
        )

        if done:
            truncated = info.get("TimeLimit.truncated", False)
            if truncated:
                return dm_env.truncation(reward, observation)
            return dm_env.termination(reward, observation)
        return dm_env.transition(reward, observation)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Returns the last info returned from env.step(action).

        Returns:
          info: dictionary of diagnostic information from the last environment step
        """
        return self._last_info

    @property
    def environment(self) -> gym.Env:
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self._environment, name)

    def close(self):
        self._environment.close()

    def get_task(self):
        return {
            "language_instruction": ["move pin to the hole"],
        }


def _convert_to_spec(space: gym.Space, name: Optional[str] = None):
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, spaces.Box):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
            name=name,
        )

    elif isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(
            shape=space.shape, dtype=space.dtype, minimum=0.0, maximum=1.0, name=name
        )

    elif isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=np.zeros(space.shape),
            maximum=space.nvec - 1,
            name=name,
        )

    elif isinstance(space, spaces.Tuple):
        return tuple(_convert_to_spec(s, name) for s in space.spaces)

    elif isinstance(space, spaces.Dict):
        return {
            key: _convert_to_spec(value, key) for key, value in space.spaces.items()
        }

    else:
        raise ValueError("Unexpected gym space: {}".format(space))


def convert_to_np(data):
    if isinstance(data, tf.Tensor):
        return data.numpy()
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: convert_to_np(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_np(value) for value in data]
    else:
        return data

def squeeze_all_zero(data: dict):
    for key, value in data.items():
        data[key] = value.squeeze(0)
    return data