import optuna
import logging
import numpy as np

# disable skrl logging
from skrl import logger

logger.setLevel(logging.WARNING)


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    actor_learning_rate = trial.suggest_float("actor_learning_rate", low=1e-7, high=1e-5, log=True)
    critic_learning_rate = trial.suggest_float("critic_learning_rate", low=1e-7, high=1e-5, log=True)
    discount_factor = trial.suggest_categorical("discount_factor", [0.9, 0.95, 0.98, 0.99, 0.999])
    grad_norm_clip = trial.suggest_categorical("grad_norm_clip", [1.0, 0.5, 0.3])
    polyak = trial.suggest_float("polyak", low=0.0001, high=0.001, log=True)
    entropy_learning_rate = trial.suggest_float("entropy_learning_rate", low=1e-7, high=1e-5, log=True)

    # # metrics
    episode_rewards = []
    instantaneous_rewards = []

    """
    RL
    """

    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast
    from skrl.envs.loaders.torch import load_isaaclab_env
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed
    from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
    from skrl.agents.torch.sac import SAC
    from skrl.utils.spaces.torch import unflatten_tensorized_space

    set_seed(42)


    class Actor(GaussianMixin, Model):
        def __init__(
            self,
            observation_space,
            action_space,
            device,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-5,
            max_log_std=2,
        ):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

            self.net_fc = nn.Sequential(
                nn.Linear(512 + 30, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32, self.num_actions),
            )
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            states = unflatten_tensorized_space(self.observation_space, inputs["states"])
            image = states["image"].view(-1, *self.observation_space["image"].shape)
            fusion = torch.cat([image, states["value"]], dim=1)
            fc = self.net_fc(fusion)

            return (
                fc,
                self.log_std_parameter,
                {},
            )


    class Critic(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net_mlp = nn.Sequential(
                nn.Linear(512 + 54 + self.num_actions, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32, 1),
            )

        def compute(self, inputs, role):
            states = unflatten_tensorized_space(self.observation_space, inputs["states"])
            image = states["image"].view(-1, *self.observation_space["image"].shape)
            taken_actions = inputs["taken_actions"]
            i = torch.cat([image, states["critic"], taken_actions], dim=1)
            mlp = self.net_mlp(i)

            return (
                mlp,
                {},
            )

    env = load_isaaclab_env(task_name="Isaac-AGV-Direct")
    env = wrap_env(env, wrapper="isaaclab-single-agent")

    device = env.device

    replay_buffer_size = 1024
    memory_size = int(replay_buffer_size / env.num_envs)
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)


    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["gradient_steps"] = 1
    cfg["batch_size"] = batch_size
    cfg["discount_factor"] = discount_factor
    cfg["polyak"] = polyak
    cfg["actor_learning_rate"] = actor_learning_rate
    cfg["critic_learning_rate"] = critic_learning_rate
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = grad_norm_clip
    cfg["learn_entropy"] = True
    cfg["entropy_learning_rate"] = entropy_learning_rate
    cfg["initial_entropy_value"] = 0.9

    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0

    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # agent.load("./runs/torch/AGV/24-10-11_17-43-00-133994_SAC/checkpoints/agent_400000.pt")

    cfg_trainer = {
        "timesteps": 100,
        "headless": True,
        # "disable_progressbar": True,
        "close_environment_at_exit": False,
    }

    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)


    # train the agent
    with autocast():
        trainer.train()
    
    # close the environment
    env.close()

    # ---------------------------------

    return np.mean(episode_rewards)


storage = "sqlite:///hyperparameter_optimization.db"
sampler = optuna.samplers.TPESampler()
direction = "maximize"  # maximize episode reward

study = optuna.create_study(
    storage=storage,
    sampler=sampler,
    study_name="optimization",
    direction=direction,
    load_if_exists=True,
)

study.optimize(objective, n_trials=25)

print(f"The best trial obtains a normalized score of {study.best_trial.value}", study.best_trial.params)
