import random
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
import omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp as mdp
import torch
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from omni.isaac.lab.managers import ManagerTermBase as TermBase
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, TiledCameraCfg
from omni.isaac.lab.utils import configclass
from .agv_cfg import AGV_CFG, AGV_JOINT
from pxr import Gf

K = 1e3
M = 1e6
CORRECT_DISTANCE = 0.005
ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class AGVSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),  # size=(100.0, 100.0)
    )

    agv: ArticulationCfg = AGV_CFG.replace(prim_path=f"{ENV_REGEX_NS}/AGV")

    niro = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/Niro",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sites/IsaacLab/model/niro/niro_no_joint.usd",
            activate_contact_sensors=True,
            # mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # disable_gravity=True,
                kinematic_enabled=True,
            ),
            # articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.6, 0.0, 1.05),
            # rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    # table = RigidObjectCfg(
    #     prim_path=f"{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="./robot/usd/table.usd",
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=100000.0),
    #         # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         #     articulation_enabled=False
    #         # ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0, -0.7, 1.1),
    #     ),
    # )

    rcam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/AGV/rcam_1/Camera",
        data_types=["rgb", "depth"],
        spawn=None,
        height=512,
        width=512,
    )

    # lcam: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/AGV/lcam_1/Camera",
    #     data_types=["rgb"],
    #     spawn=None,
    #     height=300,
    #     width=300,
    # )

    niro_contact = ContactSensorCfg(prim_path=f"{ENV_REGEX_NS}/Niro/de_1")
    agv_contact = ContactSensorCfg(prim_path=f"{ENV_REGEX_NS}/AGV/mb_1")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         # AGV_JOINT.MB_LW_REV,
    #         # AGV_JOINT.MB_RW_REV,
    #         AGV_JOINT.MB_PZ_PRI,
    #         AGV_JOINT.PZ_PY_PRI,
    #         AGV_JOINT.PY_PX_PRI,
    #         AGV_JOINT.PX_PR_REV,
    #         AGV_JOINT.PR_LR_REV,
    #         AGV_JOINT.PR_RR_REV,
    #         AGV_JOINT.LR_LPIN_PRI,
    #         AGV_JOINT.RR_RPIN_PRI,
    #     ],
    #     scale=100.0,
    # )

    joint_pri = mdp.JointPositionActionCfg(
        asset_name="agv",
        joint_names=[
            # AGV_JOINT.MB_PZ_PRI,
            AGV_JOINT.PZ_PY_PRI,
            AGV_JOINT.PY_PX_PRI,
        ],
        scale=1,
    )

    # joint_pxpr = mdp.JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         AGV_JOINT.PX_PR_REV,
    #     ],
    #     scale=100.0,
    # )

    # joint_rev = mdp.JointPositionActionCfg(
    #     asset_name="agv",
    #     joint_names=[
    #         AGV_JOINT.PR_LR_REV,
    #         AGV_JOINT.PR_RR_REV,
    #     ],
    #     scale=0.1,
    # )
    joint_pin = mdp.JointPositionActionCfg(
        asset_name="agv",
        joint_names=[
            # AGV_JOINT.LR_LPIN_PRI,
            AGV_JOINT.RR_RPIN_PRI,
        ],
        scale=1,
    )

    # IK imitation learning
    # PRI = mdp.JointPositionActionCfg(
    #     asset_name="agv",
    #     joint_names=[
    #         AGV_JOINT.PZ_PY_PRI,
    #         AGV_JOINT.PY_PX_PRI,
    #         AGV_JOINT.RR_RPIN_PRI,
    #     ],
    #     scale=1,
    # )


@configclass
class TheiaTinyObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("rcam"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv",
                "model_device": "cuda:0",
            },
        )

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


@configclass
class DictObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        rcam_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("rcam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        rcam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("rcam"),
                "data_type": "depth",
            },
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("agv")}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("agv")}
        )
        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: ObsGroup = PolicyCfg()


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, env_ids=env_ids
        )


def randomize_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos += math_utils.sample_uniform(
        *position_range, joint_pos.shape, joint_pos.device
    )
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, 0, env_ids=env_ids)


def randomize_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    xy_position_range: tuple[float, float],
    z_position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
):
    rigid_object = env.scene.rigid_objects[asset_cfg.name]
    # obtain default and deal with the offset for env origins
    default_root_state = rigid_object.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    xy_low, xy_high = xy_position_range
    z_low, z_high = z_position_range

    # Random offsets for X and Y coordinates
    xy_random_offsets = torch.tensor(
        np.random.uniform(
            xy_low, xy_high, size=(default_root_state.shape[0], 2)
        ),  # For X and Y only
        dtype=default_root_state.dtype,
        device=default_root_state.device,
    )

    # Random offsets for Z coordinate
    z_random_offsets = torch.tensor(
        np.random.uniform(
            z_low, z_high, size=(default_root_state.shape[0], 1)
        ),  # For Z only
        dtype=default_root_state.dtype,
        device=default_root_state.device,
    )

    # Apply random offsets to the X, Y, and Z coordinates
    default_root_state[:, 0:2] += xy_random_offsets  # Apply to X and Y coordinates
    default_root_state[:, 2:3] += z_random_offsets  # Apply to Z coordinate

    # set into the physics simulatio
    rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)


def randomize_camera_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rcam"),
):
    rigid_object = env.scene.rigid_objects[asset_cfg.name]
    default_root_state = rigid_object.data.default_root_state[env_ids].clone()
    print(env.scene.env_origins[env_ids])
    default_root_state[:, 4:] += env.scene.env_origins[env_ids]

    rot_low, rot_high = -0.1, 0.1

    rot_random_offsets = torch.tensor(
        np.random.uniform(rot_low, rot_high, size=(default_root_state.shape[0], 1)),
        dtype=default_root_state.dtype,
        device=default_root_state.device,
    )

    default_root_state[:, 2:3] += rot_random_offsets

    rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)


def reset_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("table"),
):
    rigid_object = env.scene.rigid_objects[asset_cfg.name]
    rigid_object.write_root_state_to_sim(table_state, env_ids=env_ids)


def init_table(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("table"),
):
    global table_state
    rigid_object = env.scene.rigid_objects[asset_cfg.name]
    table_state = rigid_object.data.root_link_state_w.clone()


class PinRewBase:
    def pin_positions(self, right: bool = True, env_ids=None):
        pin_idx = self._env.scene.articulations["agv"].find_bodies(
            "rpin_1" if right else "lpin_1"
        )[0]
        pin_root_pos = self._env.scene.articulations["agv"].data.body_link_pos_w[
            env_ids, pin_idx, :
        ]
        pin_rel = torch.tensor(
            [0, 0.02 if right else -0.02, 0.45], device="cuda:0"
        )  # 0.479
        pin_pos_w = torch.add(pin_root_pos, pin_rel)
        return pin_pos_w.squeeze(1)

    def hole_positions(self, right: bool = True, env_ids=None):
        niro: RigidObject = self._env.scene.rigid_objects["niro"]
        niro_pos = niro.data.root_pos_w[env_ids]
        hole_rel = torch.tensor(
            [0.455, 0.693 if right else -0.693, 0.0654], device="cuda:0"
        )
        hole_pos_w = torch.add(niro_pos, hole_rel)
        return hole_pos_w.squeeze(1)

    def pin_velocities(self, right: bool = True, env_ids=None):
        pin_idx = self._env.scene.articulations["agv"].find_bodies(
            "rpin_1" if right else "lpin_1"
        )[0]
        pin_vel_w = self._env.scene.articulations["agv"].data.body_link_vel_w[
            env_ids, pin_idx, :
        ]
        pin_lv = pin_vel_w.squeeze(1)[..., :3]
        pin_v_norm = torch.norm(pin_lv, dim=-1)
        return pin_v_norm


class NiroRewBase:
    def niro_velocities(self, env_ids=None):
        niro = self._env.scene.rigid_objects["niro"]
        niro_vel_w = niro.data.body_com_vel_w
        niro_lv = niro_vel_w.squeeze(1)  # [..., :3]
        niro_vel_norm = torch.norm(niro_lv[..., :3], dim=-1) + torch.norm(
            niro_lv[..., 3:], dim=-1
        )
        return niro_vel_norm

    def niro_accelerations(self, env_ids=None):
        niro = self._env.scene.rigid_objects["niro"]
        niro_acc_w = niro.data.body_acc_w
        niro_la = niro_acc_w.squeeze(1)  # [..., :3]
        niro_acc_norm = torch.norm(niro_la[..., :3], dim=-1) + torch.norm(
            niro_la[..., 3:], dim=-1
        )
        return niro_acc_norm


class niro_reward(TermBase, NiroRewBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewTerm):
        super().__init__(cfg, env)
        self.init_niro_vel = niro_velocity_norm(env)
        self.init_niro_acc = niro_acceleration_norm(env)

    def reset(self, env_ids: torch.Tensor):
        niro_vel_w = self.niro_velocities(env_ids)
        niro_acc_w = self.niro_accelerations(env_ids)

        self.init_niro_vel[env_ids] = niro_vel_w
        self.init_niro_acc[env_ids] = niro_acc_w

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("niro"),
    ) -> torch.Tensor:
        niro_vel_norm = niro_velocity_norm(env)
        niro_acc_norm = niro_acceleration_norm(env)

        reward = niro_vel_norm**2 + niro_acc_norm**2
        return -reward


class pin_pos_reward(TermBase, PinRewBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewTerm):
        super().__init__(cfg, env)
        self.init_pin_pos = all_pin_positions(env, cfg.params["right"])
        self.init_hole_pos = all_hole_positions(env, cfg.params["right"])
        self.init_distance = euclidean_distance(self.init_pin_pos, self.init_hole_pos)

    def reset(self, env_ids: torch.Tensor):
        pin_pos_w = self.pin_positions(self.cfg.params["right"], env_ids)
        hole_pos_w = self.hole_positions(self.cfg.params["right"], env_ids)

        self.init_distance[env_ids] = euclidean_distance(pin_pos_w, hole_pos_w)
        self.init_pin_pos[env_ids] = pin_pos_w
        self.init_hole_pos[env_ids] = hole_pos_w

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_pos_w = all_pin_positions(env, right)
        distance = euclidean_distance(pin_pos_w, self.init_hole_pos)

        curr_pin_pos_w = pin_pos_w
        curr_pin_to_hole = distance

        init_pin_to_cur_pin = euclidean_distance(self.init_pin_pos, curr_pin_pos_w)

        dist3 = init_pin_to_cur_pin + curr_pin_to_hole - self.init_distance
        rew = dist3**3

        xyz_rew = (self.init_distance - curr_pin_to_hole) ** 3

        reward = xyz_rew - rew

        # self.calculate_angles(curr_pin_pos_w, self.init_hole_pos)

        # 1111
        # pin_vel_w = all_pin_velocities(env, right)
        # pin_acc_w = all_pin_accelerations(env, right) + 1e-8
        # print(f"pin_vel_w: {pin_vel_w}")
        # print(f"pin_acc_w: {pin_acc_w}")
        # print(f"reward: {reward}")
        # reward = (
        #     reward
        #     * torch.clamp(1/pin_vel_w, min=1, max=10)
        #     * torch.clamp(1/pin_acc_w, min=1, max=10)
        # )

        return reward

    def calculate_angles(self, a, b):
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        u = a - b
        dot_product = torch.sum(u * z_axis, dim=1)
        norm_u = torch.norm(u, dim=1)
        norm_z = torch.norm(z_axis)
        cos_theta = dot_product / (norm_u * norm_z + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta_radian = torch.acos(cos_theta)
        angles_degree = torch.rad2deg(theta_radian)

        return angles_degree


class pin_pos_xy_reward(pin_pos_reward):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_pos_w = all_pin_positions(env, right)
        init_distances = euclidean_distance(
            self.init_pin_pos[:, :2], self.init_hole_pos[:, :2]
        )
        curr_distances = euclidean_distance(pin_pos_w[:, :2], self.init_hole_pos[:, :2])
        return (init_distances - curr_distances) ** 2


class pin_pos_z_reward(pin_pos_reward):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_pos_w = all_pin_positions(env, right)
        init_distances = torch.abs(self.init_pin_pos[:, 2] - self.init_hole_pos[:, 2])
        curr_distances = pin_pos_w[:, 2] - self.init_hole_pos[:, 2]
        return (init_distances - curr_distances) ** 2


class pin_vel_reward(TermBase, PinRewBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        target_velocity: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_vel_w = all_pin_velocities(env, right)
        return pin_vel_w**2


class pin_acc_reward(TermBase, PinRewBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_acc_w = all_pin_accelerations(env, right)
        return pin_acc_w**2


class pin_torque_reward(TermBase, PinRewBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        joint_idx = self._env.scene.articulations["agv"].find_joints(
            [
                AGV_JOINT.RR_RPIN_PRI if right else AGV_JOINT.LR_LPIN_PRI,
                AGV_JOINT.PZ_PY_PRI,
                AGV_JOINT.PY_PX_PRI,
            ]
        )[0]
        torques = self._env.scene.articulations["agv"].data.applied_torque[:, joint_idx]
        return torch.sum(torques**2, dim=1)


class decalcomanie_reward(TermBase, PinRewBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        joint_idx = self._env.scene.articulations["agv"].find_joints(
            [
                # AGV_JOINT.PR_RR_REV,
                AGV_JOINT.RR_RPIN_PRI,
                # AGV_JOINT.PR_LR_REV,
                AGV_JOINT.LR_LPIN_PRI,
            ]
        )[0]  # [6, 7, 8, 9]
        position = self._env.scene.articulations["agv"].data.joint_pos[:, joint_idx]
        L = position[:, 0]
        R = position[:, 1]
        DIFF = torch.abs(L - R)
        return torch.sum(DIFF**2)


class pin_direction_penalty(TermBase, PinRewBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewTerm):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, device="cuda:0")
        self.prev_direction = 0
        self.prev_pin_pos = all_pin_positions(env, cfg.params["right"])

    def reset(self, env_ids: torch.Tensor):
        pin_pos_w = self.pin_positions(self.cfg.params["right"], env_ids)

        self.prev_pin_pos[env_ids] = pin_pos_w
        self.counter = torch.zeros_like(self.counter)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        curr_pin_pos = all_pin_positions(env, right)
        curr_z = curr_pin_pos[..., 2]
        prev_z = self.prev_pin_pos[..., 2]
        direction = torch.where(curr_z > prev_z, 1, -1)

        self.counter = torch.where(
            direction == self.prev_direction, self.counter + 1, 0
        )
        self.prev_direction = direction

        return self.counter**2


class draw_lines(TermBase):
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        pin_list = torch.vstack(
            (all_pin_positions(env, True), all_pin_positions(env, False))
        ).tolist()
        hole_list = torch.vstack(
            (all_hole_positions(env, True), all_hole_positions(env, False))
        ).tolist()
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        draw.draw_lines(
            pin_list,
            hole_list,
            [(1, 1, 1, 1)] * self.num_envs * 2,
            [5] * self.num_envs * 2,
        )
        return torch.zeros(self.num_envs, device="cuda:0").bool()


class pin_correct_reward(TermBase, PinRewBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewTerm):
        super().__init__(cfg, env)
        self.init_hole_pos = all_hole_positions(env, cfg.params["right"])
        self.init_pin_pos = all_pin_positions(env, cfg.params["right"])

    def reset(self, env_ids: torch.Tensor):
        self.init_hole_pos[env_ids] = self.hole_positions(
            self.cfg.params["right"], env_ids
        )
        self.init_pin_pos[env_ids] = self.pin_positions(
            self.cfg.params["right"], env_ids
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        # pin_pos = all_pin_positions(env, right)
        pin_vel = all_pin_velocities(env, right)

        # pin_xy_pos = pin_pos[..., :2]
        # pin_z_pos = pin_pos[..., 2]
        # hole_xy_pos = self.init_hole_pos[..., :2]
        # hole_z_pos = self.init_hole_pos[..., 2]

        # xy_distance = euclidean_distance(hole_xy_pos, pin_xy_pos)
        # xy_correct = xy_distance < 0.01
        # z_correct = torch.logical_and(hole_z_pos + 0.02 > pin_z_pos, pin_z_pos > hole_z_pos - 0.01)

        # distance = euclidean_distance(self.init_hole_pos, pin_pos)
        # pos_correct = torch.logical_and(xy_correct, z_correct)

        # reward = pos_correct.int() * torch.clamp(1 / pin_vel, max=10) * torch.clamp(1 / distance, max=10)
        reward = pin_correct(env, right).int() * torch.clamp(
            1 / pin_vel, max=10, min=0.1
        )
        return reward


class pin_wrong_reward(TermBase, PinRewBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewTerm):
        super().__init__(cfg, env)
        self.init_hole_pos = all_hole_positions(env, cfg.params["right"])

    def reset(self, env_ids: torch.Tensor):
        self.init_hole_pos[env_ids] = self.hole_positions(
            self.cfg.params["right"], env_ids
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        right: bool = True,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("agv"),
    ) -> torch.Tensor:
        penalty = pin_wrong(env, right)
        pin_pos = all_pin_positions(env, right)
        reward = penalty.int() * euclidean_distance(self.init_hole_pos, pin_pos)
        return reward**2


def randomize_color(env: ManagerBasedEnv, env_ids: torch.Tensor):
    object_names = ["AGV", "Niro"]
    material_names = ["OmniSurfaceLite", "material_silver"]
    property_names = [
        "Shader.inputs:diffuse_reflection_color",
        "Shader.inputs:diffuse_color_constant",
    ]
    stage = stage_utils.get_current_stage()

    for idx, object_name in enumerate(object_names):
        for env_id in env_ids:
            color = Gf.Vec3f(random.random(), random.random(), random.random())
            color_spec = stage.GetAttributeAtPath(
                f"/World/envs/env_{env_id}/{object_name}/Looks/{material_names[idx]}/{property_names[idx]}"
            )
            color_spec.Set(color)


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_xyz_position = EventTerm(
        func=randomize_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "agv",
                joint_names=[
                    # AGV_JOINT.MB_PZ_PRI,
                    AGV_JOINT.PZ_PY_PRI,
                    AGV_JOINT.PY_PX_PRI,
                ],
            ),
            "position_range": (-0.05, 0.05),
        },
    )

    reset_pin_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "agv",
                joint_names=[
                    # AGV_JOINT.LR_LPIN_PRI,
                    AGV_JOINT.RR_RPIN_PRI,
                ],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    # reset_rev_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[AGV_JOINT.PX_PR_REV]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )

    randomize_color = EventTerm(
        func=randomize_color,
        mode="reset",
    )

    reset_niro_position = EventTerm(
        func=randomize_object_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("niro"),
            "xy_position_range": (-0.05, 0.05),
            "z_position_range": (-0.05, 0.05),
        },
    )

    # reset_table_position = EventTerm(
    #     func=reset_object_position,
    #     mode="reset",
    # )

    # init_table = EventTerm(
    #     func=init_table,
    #     mode="startup",
    # )

    # random_camera_rotation = EventTerm(
    #     func=randomize_camera_position,
    #     mode="reset",
    # )


def all_pin_positions(env: ManagerBasedRLEnv, right: bool = True):
    pin_idx = env.scene.articulations["agv"].find_bodies(
        "rpin_1" if right else "lpin_1"
    )[0]
    pin_root_pos = env.scene.articulations["agv"].data.body_link_pos_w[:, pin_idx, :]
    pin_rel = torch.tensor(
        [0, 0.02 if right else -0.02, 0.45], device="cuda:0"
    )  # 0.479

    angle_deg = 40
    angle_rad = (angle_deg if right else -angle_deg) * (np.pi / 180.0)
    cos_theta = torch.cos(torch.tensor(angle_rad, device="cuda:0"))
    sin_theta = torch.sin(torch.tensor(angle_rad, device="cuda:0"))

    rotation_matrix = torch.tensor(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
        device="cuda:0",
    )

    pin_rel_rotated = torch.matmul(rotation_matrix, pin_rel)

    pin_pos_w = torch.add(pin_root_pos, pin_rel_rotated)
    return pin_pos_w.squeeze(1)


def all_hole_positions(env: ManagerBasedRLEnv, right: bool = True):
    niro: RigidObject = env.scene.rigid_objects["niro"]
    niro_pos = niro.data.root_pos_w
    hole_rel = torch.tensor(
        [0.455, 0.693 if right else -0.693, 0.0654], device="cuda:0"
    )
    hole_pos_w = torch.add(niro_pos, hole_rel)
    return hole_pos_w.squeeze(1)


def niro_velocity_norm(env: ManagerBasedRLEnv):
    niro: RigidObject = env.scene.rigid_objects["niro"]
    niro_lv = niro.data.body_com_vel_w.squeeze(1)
    niro_vel_norm = torch.norm(niro_lv[..., :3], dim=-1) + torch.norm(
        niro_lv[..., 3:], dim=-1
    )

    return niro_vel_norm


def niro_acceleration_norm(env: ManagerBasedRLEnv):
    niro: RigidObject = env.scene.rigid_objects["niro"]
    niro_la = niro.data.body_acc_w.squeeze(1)
    niro_acc_norm = torch.norm(niro_la[..., :3], dim=-1) + torch.norm(
        niro_la[..., 3:], dim=-1
    )
    return niro_acc_norm


def all_pin_velocities(env: ManagerBasedRLEnv, right: bool = True, env_id=None):
    pin_idx = env.scene.articulations["agv"].find_bodies(
        "rpin_1" if right else "lpin_1"
    )[0]
    # pin_idx = env.scene.articulations["agv"].find_joints(AGV_JOINT.RR_RPIN_PRI if right else AGV_JOINT.LR_LPIN_PRI)[0]
    pin_vel_w = env.scene.articulations["agv"].data.body_link_vel_w[:, pin_idx, :]

    pin_lv = pin_vel_w.squeeze(1)[..., :3]
    pin_v_norm = torch.norm(pin_lv, dim=-1)
    return pin_v_norm + 1e-8


def all_pin_accelerations(env: ManagerBasedRLEnv, right: bool = True, env_id=None):
    pin_idx = env.scene.articulations["agv"].find_bodies(
        "rpin_1" if right else "lpin_1"
    )[0]
    # pin_idx = env.scene.articulations["agv"].find_joints(AGV_JOINT.RR_RPIN_PRI if right else AGV_JOINT.LR_LPIN_PRI)[0]
    pin_vel_w = env.scene.articulations["agv"].data.body_acc_w[:, pin_idx, :]

    pin_lv = pin_vel_w.squeeze(1)[..., :3]
    pin_v_norm = torch.norm(pin_lv, dim=-1)
    return pin_v_norm


def euclidean_distance(src, dist):
    distance = torch.sqrt(torch.sum((src - dist) ** 2, dim=src.ndim - 1) + 1e-8)
    return distance


def power_reward(reward) -> torch.Tensor:
    return torch.where(reward < 0, -((reward - 1) ** 2), (reward + 1) ** 2)


class power_consumption(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"],
            asset.joint_names,
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg,
        right: bool = True,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        joints = [
            AGV_JOINT.RR_RPIN_PRI if right else AGV_JOINT.LR_LPIN_PRI,
            AGV_JOINT.PY_PX_PRI,
            AGV_JOINT.PZ_PY_PRI,
        ]
        idx = asset.find_joints(joints)[0]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(
            torch.abs(
                env.action_manager.action
                * asset.data.joint_vel[:, idx]
                * self.gear_ratio_scaled[:, idx]
            ),
            dim=-1,
        )


@configclass
class RewardsCfg:
    # alive = RewTerm(func=mdp.is_alive, weight=1)
    # terminating = RewTerm(func=mdp.is_terminated, weight=-1000)

    # decalcomanie = RewTerm(func=decalcomanie_reward, weight=-100)

    r_pin_pos = RewTerm(func=pin_pos_reward, weight=10000, params={"right": True})
    # l_pin_pos = RewTerm(func=pin_pos_reward, weight=10000, params={"right": False})
    # r_pin_xy_pos = RewTerm(func=pin_pos_xy_reward, weight=-1000, params={"right": True})
    # l_pin_xy_pos = RewTerm(func=pin_pos_xy_reward, weight=-1000, params={"right": False})
    # r_pin_z_pos = RewTerm(func=pin_pos_z_reward, weight=-50, params={"right": True})
    # l_pin_z_pos = RewTerm(func=pin_pos_z_reward, weight=-50, params={"right": False})
    r_pin_vel = RewTerm(func=pin_vel_reward, weight=-10, params={"right": True})
    # l_pin_vel = RewTerm(func=pin_vel_reward, weight=-10, params={"right": False})
    r_pin_acc = RewTerm(func=pin_acc_reward, weight=-1e-2, params={"right": True})
    # l_pin_acc = RewTerm(func=pin_acc_reward, weight=-1e-2, params={"right": False})
    # r_pin_joint_acc = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-1e-4,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "agv",
    #             joint_names=[AGV_JOINT.RR_RPIN_PRI],
    #         )
    #     },
    # )
    # r_pin_joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "agv",
    #             joint_names=[AGV_JOINT.RR_RPIN_PRI],
    #         )
    #     },
    # )
    # r_pin_torque = RewTerm(func=pin_torque_reward, weight=-1e-6, params={"right": True})
    r_pin_joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "agv",
                joint_names=[AGV_JOINT.RR_RPIN_PRI],
            )
        },
    )
    # l_pin_joint_torque = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-2e-3,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "agv",
    #             joint_names=[AGV_JOINT.LR_LPIN_PRI],
    #         )
    #     },
    # )

    # xy_joint_acc = RewTerm(
    #     func=mdp.joint_acc_l2,
    #     weight=-0.01,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "agv",
    #             joint_names=[AGV_JOINT.PZ_PY_PRI, AGV_JOINT.PY_PX_PRI],
    #         )
    #     },
    # )
    # xy_joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-100,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "agv",
    #             joint_names=[AGV_JOINT.PZ_PY_PRI, AGV_JOINT.PY_PX_PRI],
    #         )
    #     },
    # )
    xy_joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "agv",
                joint_names=[AGV_JOINT.PZ_PY_PRI, AGV_JOINT.PY_PX_PRI],
            )
        },
    )
    r_pin_correct = RewTerm(func=pin_correct_reward, weight=10, params={"right": True})
    # l_pin_correct = RewTerm(func=pin_correct_reward, weight=10, params={"right": False})

    # agv_undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("agv_contact"),
    #         "threshold": 1.0,
    #     },
    # )

    # r_pin_wrong = RewTerm(func=pin_wrong_reward, weight=-3e3, params={"right": True})
    # l_pin_wrong = RewTerm(func=pin_wrong_reward, weight=-3e3, params={"right": False})

    niro_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10,
        params={
            "sensor_cfg": SceneEntityCfg("niro_contact"),
            "threshold": 0,
        },
    )

    # niro_contact_force = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("niro_contact"),
    #         "threshold": 0,
    #     },
    # )

    # niro_vel_acc = RewTerm(func=niro_reward, weight=0.1)


def pin_correct(env, right: bool = True) -> torch.Tensor:
    hole_pos_w = all_hole_positions(env, right)
    pin_pos_w = all_pin_positions(env, right)
    distance = euclidean_distance(hole_pos_w, pin_pos_w)

    pin_pos = distance < CORRECT_DISTANCE  # 0.01 -> 0.005 -> 0.001
    # pin_vel = all_pin_velocities(env, right) < 0.01
    # pin_correct = torch.logical_and(pin_pos, pin_vel)

    return pin_pos.squeeze(0)


def pin_wrong(env, right: bool = True) -> torch.Tensor:
    hole_pos_w = all_hole_positions(env, right)
    pin_pos_w = all_pin_positions(env, right)

    hole_xy = hole_pos_w[:, :2]
    pin_xy = pin_pos_w[:, :2]
    xy_distance = euclidean_distance(hole_xy, pin_xy)

    hole_z = hole_pos_w[:, 2]
    pin_z = pin_pos_w[:, 2]

    z_condition = pin_z >= hole_z - 0.1
    xy_condition = xy_distance >= 0.01

    return torch.logical_and(z_condition, xy_condition)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # niro_bad_orientation = DoneTerm(
    #     func=termination_accel,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "niro",
    #             # joint_names=[AGV_JOINT.MB_PZ_PRI],
    #         ),
    #         "limit_acc": 1.0,
    #     },
    # )

    # pole_out_of_bounds = DoneTerm(func=pin_wrong, params={"right": True})
    # pole_contacts = DoneTerm(func=undesired_contacts)

    pin_correct = DoneTerm(func=pin_correct, params={"right": True})
    draw_lines = DoneTerm(func=draw_lines)


def modify_correct_distance(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], distance: float, num_steps: int
):
    if env.common_step_counter > num_steps:
        global CORRECT_DISTANCE
        CORRECT_DISTANCE = distance


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    # 1M
    # r_pin_correct = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "r_pin_correct", "weight": 1, "num_steps": 5 * K},
    # )
    # r_pin_wrong_10K = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "r_pin_wrong", "weight": -1e4, "num_steps": 10 * K},
    # )

    # 2M
    # niro_undesired_contacts = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "niro_undesired_contacts", "weight": -2, "num_steps": 10 * K},
    # )

    # 3M
    # distance_5mm = CurrTerm(
    #     func=modify_correct_distance,
    #     params={"distance": 0.003, "num_steps": 20 * K},
    # )
    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "joint_vel", "weight": -1, "num_steps": 15 * K},
    # )
    # joint_torque = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "joint_torque", "weight": -5e-4, "num_steps": 15 * K},
    # )
    # joint_torque_50K = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "joint_torque", "weight": -1e-5, "num_steps": 50 * K},
    # )
    # 5M
    # distance_3mm = CurrTerm(
    #     func=modify_correct_distance,
    #     params={"distance": 0.003, "num_steps": 30 * K},
    # )
    # distance_1mm = CurrTerm(
    #     func=modify_correct_distance,
    #     params={"distance": 0.001, "num_steps": 50 * K},
    # )
    # r_pin_wrong_50K = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "r_pin_wrong", "weight": -1e5, "num_steps": 20 * K},
    # )

    # 10M
    # r_pin_direction = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "r_pin_direction", "weight": -0.001, "num_steps": 40 * K},
    # )


@configclass
class AGVEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: AGVSceneCfg = AGVSceneCfg(num_envs=4, env_spacing=3.0)
    # Basic settings
    observations = DictObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # self.viewer.lookat = (0.0, 0.0, 2.5)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.render.enable_translucency = True
        self.sim.render.enable_reflections = True
        # self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_dlssg = True
        self.sim.render.dlss_mode = 3
        self.sim.render.enable_direct_lighting = True
        self.sim.render.enable_shadows = True
        self.sim.render.enable_ambient_occlusion = True
