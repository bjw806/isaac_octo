import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task", type=str, default="Isaac-Station-Managed", help="Name of the task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import carb

from util.se3_keyboard_station import Se3KeyboardAGV
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.terminations.time_out = None
    env = gym.make(args_cli.task, cfg=env_cfg)

    teleop_interface = Se3KeyboardAGV(pos_sensitivity=0.02)
    teleop_interface.add_callback("L", env.reset)
    print(teleop_interface)

    env.reset()
    teleop_interface.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            delta_pose = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            actions = delta_pose
            _, _, terminated, truncated, _ = env.step(actions)

            if terminated.any() or truncated.any():
                env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
