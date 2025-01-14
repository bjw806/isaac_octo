from typing import Any, Tuple, Union

import gymnasium
import jax.numpy as jnp
import torch
import numpy as np
from skrl.envs.wrappers.torch.base import Wrapper
from collections import deque
from .gym_wrappers import stack_and_pad, space_stack, listdict2dictlist


class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)

        self._reset_once = True
        self._observations = None
        self._info = {}
        self._episode_length = 0

    @property
    def state_space(self) -> Union[gymnasium.Space, None]:
        try:
            return self._unwrapped.single_observation_space["critic"]
        except KeyError:
            pass
        try:
            return self._unwrapped.state_space
        except AttributeError:
            return None

    @property
    def observation_space(self) -> gymnasium.Space:
        try:
            print(self._unwrapped.single_observation_space["policy"])
            return self._unwrapped.single_observation_space["policy"]
        except:
            return self._unwrapped.observation_space["policy"]

    @property
    def action_space(self) -> gymnasium.Space:
        try:
            return self._unwrapped.single_action_space
        except:
            return self._unwrapped.action_space

    def step(
        self, actions#: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        actions = torch.from_numpy(np.array(actions)).unsqueeze(0)
        observations, reward, terminated, truncated, self._info = self._env.step(
            actions
        )
        self._observations = self.convert_to_octo(observations["policy"])
        self._episode_length += 1
        return (
            self._observations,
            reward.cpu().item(),
            # reward.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            self._info,
        )

    def reset(self, seed=None, options=None) -> Tuple[torch.Tensor, Any]:
        if self._reset_once:
            observations, self._info = self._env.reset()
            self._observations = self.convert_to_octo(observations["policy"])
            self._reset_once = False

        self._episode_length = 0
        return self._observations, self._info

    def render(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        self._env.close()

    def get_task(self) -> dict:
        # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
        task = {
            "language_instruction": "First, move the pin to the hole's XY coordinate only in the XY plane. Then, raise the Z-axis and plug the pin into the hole.",
            # "goal": {"image_primary": ""},
        }
        return task

    def convert_to_octo(self, obs):
        octo_obs = {
            "image_primary": obs["rcam_rgb"].squeeze(0),
            "depth_primary": obs["rcam_depth"].squeeze(0),
            "proprio": obs["joint_pos"].squeeze(0),
            # "task_completed": torch.tensor([[False] * 50], device=device),
            # "pad_mask_dict/proprio": torch.tensor([True], device=device),
            # "pad_mask_dict/timestep": torch.tensor([True], device=device),
            # "pad_mask_dict/image_primary": torch.tensor([True], device=device),
            # "timestep": torch.tensor([self._episode_length], device=device),
            # "timestep_pad_mask": torch.tensor([True], device=device),
            # 'timestep_pad_mask': array([[1.]])
        }

        octo_obs = self.convert_to_jax(octo_obs)

        """
        obs = {
            "image_primary": array(
                [
                    [
                        [
                            [
                                [67, 35, 115],
                                [67, 35, 115],
                                [66, 35, 114],
                                ...,
                                [194, 194, 194],
                                [194, 194, 194],
                                [194, 194, 194],
                            ],
                            ...,

                            [
                                [83, 82, 82],
                                [83, 83, 83],
                                [82, 82, 82],
                                ...,
                                [194, 194, 194],
                                [194, 194, 194],
                                [194, 194, 194],
                            ],
                        ]
                    ]
                ],
                dtype=uint8,
            ),
            "pad_mask_dict": {
                "image_primary": array([[True]]),
                "proprio": array([[True]]),
                "timestep": array([[True]]),
            },
            "proprio": array(
                [
                    [
                        [
                            -5.9872698e-02,
                            -5.9416953e-02,
                            2.0818363e-04,
                            5.4983515e-01,
                            -6.1673707e-01,
                            -9.2796572e-03,
                            -9.9979854e-01,
                            9.9387419e-01,
                            1.2705543e-03,
                            4.9221998e-01,
                        ]
                    ]
                ],
                dtype=float32,
            ),
            "task_completed": array(
                [
                    [
                        [
                            False,
                            False,
                            ...
                            False,
                            False,
                        ]
                    ]
                ]
            ),
            "timestep": array([[231]], dtype=int32),
            "timestep_pad_mask": array([[True]]),
        }
        """

        return octo_obs

    def convert_to_jax(self, data):
        if isinstance(data, dict):
            return {k: self.convert_to_jax(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_to_jax(v) for v in data]
        elif isinstance(data, torch.Tensor):
            return jnp.array(data.cpu().numpy())
        elif isinstance(data, np.ndarray):
            return jnp.array(data)
        return data


class HistoryWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, horizon: int):
        super().__init__(env)
        self.horizon = horizon

        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        self.observation_space = space_stack(self.env.observation_space, self.horizon)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(obs)
        assert len(self.history) == self.horizon
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, info


class RHCWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, exec_horizon: int):
        super().__init__(env)
        self.exec_horizon = exec_horizon

    def step(self, actions):
        if self.exec_horizon == 1 and len(actions.shape) == 1:
            actions = actions[None]
        assert len(actions) >= self.exec_horizon
        rewards = []
        observations = []
        infos = []

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            infos.append(info)

            if done or trunc:
                break

        infos = listdict2dictlist(infos)
        infos["rewards"] = rewards
        infos["observations"] = observations

        return obs, np.sum(rewards), done, trunc, infos
