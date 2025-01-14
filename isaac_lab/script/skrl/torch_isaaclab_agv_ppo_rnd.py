import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from ppo_rnd import PPO_RND, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space
from torch.cuda.amp import autocast
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
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_features = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.net_fc = nn.Sequential(
            nn.Linear(512 + 30, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.2),
        )

        self.rnd_target = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.rnd_predictor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
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
        i = torch.cat([features, states["value"]], dim=1)

        rnd_target_output = self.rnd_target(features.detach())
        rnd_predictor_output = self.rnd_predictor(features)
        rnd_bonus = F.mse_loss(rnd_predictor_output, rnd_target_output, reduction="none").mean(dim=1, keepdim=True) ** 2
        rnd_loss = F.mse_loss(rnd_predictor_output, rnd_target_output)

        if role == "policy":
            self._shared_output = self.net_fc(i)
            action = self.mean_layer(self._shared_output)
            return action, self.log_std_parameter, {"rnd_bonus": rnd_bonus, "rnd_loss": rnd_loss}
        elif role == "value":
            shared_output = self.net_fc(i) if self._shared_output is None else self._shared_output
            self._shared_output = None
            value = self.value_layer(shared_output)
            return value, {"rnd_bonus": rnd_bonus, "rnd_loss": rnd_loss}


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
env = wrap_env(env, wrapper="isaaclab-single-agent")

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
replay_buffer_size = 1024 * 1 * env.num_envs
memory_size = int(replay_buffer_size / env.num_envs)
memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = memory_size
cfg["learning_epochs"] = 64
cfg["mini_batches"] = 4
cfg["discount_factor"] = 0.99
# cfg["lambda"] = 0.95
cfg["learning_rate"] = 0
# cfg["grad_norm_clip"] = 0.5
# cfg["ratio_clip"] = 0.2
# cfg["value_clip"] = 0.2
# cfg["clip_predicted_values"] = False
# cfg["entropy_loss_scale"] = 0.5
# cfg["value_loss_scale"] = 1.0
# cfg["mixed_precision"] = False
# cfg["optimizer"] = torch.optim.Adam(models["policy"].parameters(), lr=0)
cfg["learning_starts"] = 0
cfg["learning_rate_scheduler"] = CosineAnnealingWarmUpRestarts
cfg["learning_rate_scheduler_kwargs"] = {
    "T_0": 1024,
    "T_mult": 1,
    "T_up": 64,
    "eta_max": 1e-3,
    "gamma": 0.5,
}

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 100000
cfg["experiment"]["directory"] = "runs/torch/AGV"

agent = PPO_RND(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# agent.load("./runs/torch/AGV/24-09-25_17-12-11-556727_PPO/checkpoints/agent_100000.pt")

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()
