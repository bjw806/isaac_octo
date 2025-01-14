import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect demonstrations for Isaac Lab environments."
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=1,
    help="Number of episodes to store in the dataset.",
)
parser.add_argument(
    "--filename", type=str, default="hdf_dataset", help="Basename of output file."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import tensorflow_datasets as tfds
import tensorflow as tf

from util.dm_wrapper import GymWrapper
from util.se3_keyboard_agv import Se3KeyboardAGV

import omni.isaac.lab_tasks  # noqa: F401

# from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
import envlogger
from envlogger.backends import tfds_backend_writer

task = "Isaac-AGV-Managed"
trajectories_dir = "mnt/dataset/"
num_envs = 1

def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    # parse configuration
    env_cfg = parse_env_cfg(
        task,
        device=args_cli.device,
        num_envs=num_envs,
    )

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    # env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    # env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    # env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment
    env = gym.make(task, cfg=env_cfg)
    env = GymWrapper(env)

    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name="agv_example",
        observation_info=tfds.features.FeaturesDict(
            {
                "rcam_rgb": tfds.features.Image(
                    # shape=(num_envs, 300, 300, 3),
                    shape=(
                        512,
                        512,
                        3,
                    ),
                    dtype=tf.uint8,
                ),
                "rcam_depth": tfds.features.Tensor(
                    shape=(
                        512,
                        512,
                        1,
                    ),
                    dtype=tf.float32,
                ),
                "joint_pos": tfds.features.Tensor(
                    # shape=(num_envs, 10),
                    shape=(10,),
                    dtype=tf.float32,
                ),
                "joint_vel": tfds.features.Tensor(
                    shape=(10,),
                    dtype=tf.float32,
                ),
            }
        ),
        action_info=tfds.features.Tensor(
            # shape=(num_envs, 3),
            shape=(3,),
            dtype=tf.float64,
        ),
        reward_info=tfds.features.Scalar(dtype=tf.float64),
        discount_info=tfds.features.Scalar(dtype=tf.float64),
    )

    env = envlogger.EnvLogger(
        env,
        backend=tfds_backend_writer.TFDSBackendWriter(
            data_directory=trajectories_dir,
            split_name="train",
            max_episodes_per_file=5,
            ds_config=dataset_config,
        ),
    )

    teleop_interface = Se3KeyboardAGV()

    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper
    print(teleop_interface)

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt):
        for i in range(args_cli.num_demos):
            timestep = env.reset()
            teleop_interface.reset()
            print(f"Collecting demonstration: {i + 1}/{args_cli.num_demos}")

            while not timestep.last():
                # get keyboard command
                delta_pose = teleop_interface.advance()
                actions = delta_pose

                # perform action on environment
                timestep = env.step(actions)

            print("saving...")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
