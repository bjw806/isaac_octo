from functools import partial

import jax
import numpy as np
import wandb
from absl import app, logging
from skrl.envs.loaders.torch import load_isaaclab_env

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.isaac_lab_wrapper import IsaacLabWrapper
from octo.utils.train_callbacks import supply_rng

finetuned_path = "octo/mnt/train/"


def main(_):
    # setup wandb for logging
    wandb.init(name="eval_agv", project="octo")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(finetuned_path)

    # make gym environment
    env = load_isaaclab_env(
        task_name="Isaac-AGV-Managed",
        cli_args=["--enable_cameras"],
    )
    env = IsaacLabWrapper(env)
    # wrap env to normalize proprio
    env = NormalizeProprio(env, model.dataset_statistics)
    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
            # timestep_pad_mask = [True],
        ),
    )

    # running rollouts
    for _ in range(10):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction)

        # run rollout for 400 steps
        images = [obs["image_primary"][0]]
        episode_return = 0.0
        while len(images) < 1000:
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            images.extend([o["image_primary"][0] for o in info["observations"]])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        )


if __name__ == "__main__":
    app.run(main)
