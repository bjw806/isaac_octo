import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space
from lr_schedulers import CosineAnnealingWarmUpRestarts

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed
# torch.autograd.set_detect_anomaly(True)


class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )
        DeterministicMixin.__init__(self, clip_actions)

        self.net_features = nn.Sequential(
            nn.Linear(192, 16),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.net_values = nn.Sequential(
            nn.Linear(23, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.net_fc = nn.Sequential(
            nn.Linear(324 * 16 + 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(32, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        image = states["image"].view(-1, *self.observation_space["image"].shape)
        features = self.net_features(image)
        joint = torch.cat(
            [states["joint_pos"], states["joint_vel"], states["actions"]], dim=1
        )
        values = self.net_values(joint)
        i = torch.cat([features, values], dim=1)

        if role == "policy":
            self._shared_output = self.net_fc(i)
            action = self.mean_layer(self._shared_output)
            return action, self.log_std_parameter, {}
        elif role == "value":
            # i = self.net_mlp(states["critic"])
            # i = torch.cat([image, states["critic"], taken_actions], dim=1)
            shared_output = (
                self.net_fc(i) if self._shared_output is None else self._shared_output
            )
            self._shared_output = None
            value = self.value_layer(shared_output)
            return value, {}


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Managed")
env = wrap_env(env, wrapper="isaaclab-single-agent")

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
replay_buffer_size = 512 * 1 * env.num_envs
memory_size = int(replay_buffer_size / env.num_envs)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)
# memory.load("skrl_test/memory/24-10-31_13-05-25-860930_memory_0x77a9a9ff3730.csv")

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]

# initialize models' parameters (weights and biases)
# for model in models.values():
#     model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = memory_size
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.99
# cfg["lambda"] = 0.95
cfg["learning_rate"] = 0  # CosineAnnealingWarmUpRestarts
# cfg["grad_norm_clip"] = 1.0  # gradient clipping
# cfg["ratio_clip"] = 0.1  # 정책 클리핑 (정책이 학습 초기에 과하게 수렴되지 않도록 작은 값을 설정)
# cfg["value_clip"] = 0.2
# cfg["clip_predicted_values"] = False
# cfg["entropy_loss_scale"] = 0.05
# cfg["value_loss_scale"] = 1.0
# cfg["mixed_precision"] = False
# cfg["optimizer"] = torch.optim.Adam(models["policy"].parameters(), lr=0)
cfg["learning_starts"] = 0
cfg["learning_rate_scheduler"] = CosineAnnealingWarmUpRestarts
cfg["learning_rate_scheduler_kwargs"] = {
    "T_0": 16 * cfg["learning_epochs"],  # 첫 주기의 길이
    "T_mult": 2,  # 매 주기마다 주기의 길이를 두배로 늘림
    "T_up": cfg["learning_epochs"],  # warm-up 주기
    "eta_max": 1e-3,  # 최대 학습률
    "gamma": 0.6,  # 학습률 감소율
}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1024
cfg["experiment"]["checkpoint_interval"] = 100000
cfg["experiment"]["directory"] = "runs/torch/AGV"

agent = PPO(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# agent.load("./runs/torch/AGV/24-10-30_17-27-56-597455_PPO/checkpoints/agent_400000.pt")

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 10000000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()
