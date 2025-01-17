import random
from collections.abc import Sequence
from dataclasses import MISSING
import numpy as np
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prim_utils
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

ENV_REGEX_NS = "/World/envs/env_.*"

from omni.isaac.dynamic_control import _dynamic_control



@configclass
class StationSceneCfg(InteractiveSceneCfg):
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(),  # size=(100.0, 100.0)
    # )

    station = AssetBaseCfg(
        prim_path=f"{ENV_REGEX_NS}/station",
        spawn=sim_utils.UsdFileCfg(usd_path="/home/sites/IsaacLab/model/station/station_4.usd"),

    )

    # agv = prim_utils.get_prim_at_path(f"{ENV_REGEX_NS}/station/agv_3")
    # dc = _dynamic_control.acquire_dynamic_control_interface()
    # prim = dc.get_rigid_body(f"{ENV_REGEX_NS}/station/agv_3")
    agv: ArticulationCfg = AGV_CFG.replace(
        prim_path=f"{ENV_REGEX_NS}/AGV",
        # prim_path=f"/World/station_1/agv_0",
        # spawn=None,
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sites/IsaacLab/model/agv/agv_3.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(18.0, 3.0, 1.302),
            rot=(0.707, 0.0, 0.0, -0.707),
        )
    )

    # agv:AssetBaseCfg  = AssetBaseCfg(
    #     prim_path=f"{ENV_REGEX_NS}/station/agv_3",
    #     spawn=None,
    # )

    # niro = RigidObjectCfg(
    #     prim_path=f"{ENV_REGEX_NS}/Niro",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path="/home/sites/IsaacLab/model/niro/niro_no_joint.usd",
    #         activate_contact_sensors=True,
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         # articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(-0.6, 0.0, 1.55),
    #         # rot=(0.70711, 0.0, 0.0, 0.70711),
    #     ),
    # )

    # lights
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    # )
    # distant_light = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    #     init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    agv = mdp.JointPositionActionCfg(
        asset_name="agv",
        joint_names=[
            AGV_JOINT.MB_LW_REV,
            AGV_JOINT.MB_RW_REV,
            # AGV_JOINT.MB_PZ_PRI,
            # AGV_JOINT.PZ_PY_PRI,
            # AGV_JOINT.PY_PX_PRI,
            # AGV_JOINT.PX_PR_REV,
            # AGV_JOINT.PR_LR_REV,
            # AGV_JOINT.PR_RR_REV,
            # AGV_JOINT.LR_LPIN_PRI,
            # AGV_JOINT.RR_RPIN_PRI,
        ],
        scale=1.0,
    )


@configclass
class DictObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # rcam_rgb = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("rcam"),
        #         "data_type": "rgb",
        #         "normalize": False,
        #     },
        # )

        # rcam_depth = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("rcam"),
        #         "data_type": "depth",
        #     },
        # )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("agv")}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("agv")}
        )
        actions = ObsTerm(func=mdp.last_action)

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


def reset_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("table"),
):
    rigid_object = env.scene.rigid_objects[asset_cfg.name]
    rigid_object.write_root_state_to_sim(table_state, env_ids=env_ids)


def reset_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    rigid_object = env.scene["station"]
    print(rigid_object)


@configclass
class EventCfg:
    # reset_xyz_position = EventTerm(
    #     func=reset_position,
    #     mode="reset",
    # )

    pass

def euclidean_distance(src, dist):
    distance = torch.sqrt(torch.sum((src - dist) ** 2, dim=src.ndim - 1) + 1e-8)
    return distance


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1)
    # terminating = RewTerm(func=mdp.is_terminated, weight=-1000)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # draw_lines = DoneTerm(func=draw_lines)


@configclass
class StationEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: StationSceneCfg = StationSceneCfg(num_envs=4, env_spacing=50)
    # Basic settings
    observations = DictObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

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
