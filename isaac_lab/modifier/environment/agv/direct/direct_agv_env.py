import random
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.lab.envs import (
    DirectRLEnv,
    DirectRLEnvCfg,
    ViewerCfg,
)
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    TiledCamera,
    TiledCameraCfg,
)
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from omni.isaac.debug_draw import _debug_draw
from PIL import Image
from pxr import Gf, UsdGeom
from ultralytics import YOLO, settings

from .agv_cfg import AGV_CFG, AGV_JOINT

settings.update({
    "runs_dir": "~/Desktop/repository/IsaacLab_Pitin/", 
    "weights_dir": "~/Desktop/repository/IsaacLab_Pitin/skrl_test/yolo/", 
    "tensorboard": False,
})

def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.02, 0.02, 0.02),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


##
# Scene definition
##

ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class AGVEnvCfg(DirectRLEnvCfg):
    # env
    dt = 1 / 240
    decimation = 2
    episode_length_s = 5.0
    action_scale = 1  # [N]
    num_channels = 4
    LSTM = True
    # events = AGVEventCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = AGV_CFG.replace(prim_path=f"{ENV_REGEX_NS}/AGV")

    # camera
    rcam: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/rcam_1/Camera",
        data_types=["rgb"],
        spawn=None,
        width=640,
        height=640,
        # colorize_instance_segmentation=True,
    )

    niro_cfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/Niro",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./robot/usd/niro/niro_fixed.usd",
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 1.05),
            # rot=(0.70711, 0.0, 0.0, 0.70711),
        ),
    )

    agv_joint: AGV_JOINT = AGV_JOINT()

    lpin_cfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/lpin_1",
        spawn=None,
    )

    rpin_cfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/rpin_1",
        spawn=None,
    )

    niro_contact_cfg = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/Niro/de_1",
    )
    agv_contact_cfg = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/AGV/mb_1",
    )

    actuated_joint_names = [
        # "jlw",
        # "jrw",
        # "jz",
        "jy",
        "jx",
        # "jr",
        # "jlr",
        # "jrr",
        # "jlpin",
        "jrpin",
    ]

    write_image_to_file = False

    # change viewer settings
    viewer = ViewerCfg(eye=(4.0, 0.0, 3.0))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)

    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #   noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    #   bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0015, operation="abs"),
    # )

    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #   noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #   bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )

    observation_space = gym.spaces.Dict(
        image=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(512,) if LSTM else (512, num_channels,),
        ),
        value=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30,)),
        # critic=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,)),
    )
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
    curriculum_learning = False

class AGVEnv(DirectRLEnv):
    cfg: AGVEnvCfg

    def __init__(self, cfg: AGVEnvCfg, render_mode: str | None = None, **kwargs):
        # print(1)
        super().__init__(cfg, render_mode, **kwargs)

        self.yolo = YOLO("./skrl_test/best.pt")
        # self.yolo_show = YOLO("./skrl_test/best.pt")

        # self._MB_LW_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_LW_REV)
        # self._MB_RW_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_RW_REV)
        # self._MB_PZ_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.MB_PZ_PRI)
        # self._PZ_PY_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PZ_PY_PRI)
        # self._PY_PX_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PY_PX_PRI)
        # self._PX_PR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PX_PR_REV)
        # self._PR_LR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PR_LR_REV)
        # self._PR_RR_REV_idx, _ = self._agv.find_joints(self.cfg.agv_joint.PR_RR_REV)
        # self._LR_LPIN_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.LR_LPIN_PRI)
        # self._RR_RPIN_PRI_idx, _ = self._agv.find_joints(self.cfg.agv_joint.RR_RPIN_PRI)
        self._RPIN_idx, _ = self._agv.find_bodies("rpin_1")
        self._LPIN_idx, _ = self._agv.find_bodies("lpin_1")
        # self._XY_PRI_idx, _ = self._agv.find_joints([self.cfg.agv_joint.PZ_PY_PRI, self.cfg.agv_joint.PY_PX_PRI])
        self.action_scale = self.cfg.action_scale
        self.joint_pos = self._agv.data.joint_pos
        self.joint_vel = self._agv.data.joint_vel

        self.init_distance_l = torch.zeros(self.num_envs, device=self.device)
        self.init_distance_r = torch.zeros(self.num_envs, device=self.device)
        self.init_pin_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_hole_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_z_distance_r = torch.zeros(self.num_envs, device=self.device)
        self.init_xy_distance_r = torch.zeros(self.num_envs, device=self.device)

        self.prev_pos_w = {
            "r_pin": torch.zeros(self.num_envs, 3, device=self.device),
            "l_pin": torch.zeros(self.num_envs, 3, device=self.device),
        }

        self.num_agv_dofs = self._agv.num_joints

        # # buffers for position targets
        self.agv_dof_targets = torch.zeros((self.num_envs, self.num_agv_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_agv_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_agv_dofs), dtype=torch.float, device=self.device)

        # # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self._agv.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # # joint limits
        joint_pos_limits = self._agv.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        self.joint_pos = self._agv.data.joint_pos
        self.joint_vel = self._agv.data.joint_vel
        
        self.serial_frames = torch.zeros(
            (
                self.num_envs, 
                512,
                self.cfg.num_channels,
            ), 
            dtype=torch.float, 
            device=self.device
        )
        
        self.idx = 1
        self.random_color = False
        self.random_pin_position = False
        self.random_hole_position = False
        self.reward_weights = {
            "rew_pin_r": 1000,
            "rew_pin_r_xy": 0,
            "rew_pin_vel": -0.1,
            "correct_xy_rew": 0,
            "correct_rew": 1,
            "z_penalty": 0,
            "contact_penalty": 0,
            "torque_penalty": -0.00001,
            "r_z_penalty": 0,
        }

    def close(self):
        # print(2)
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        # print(4)
        self._agv = Articulation(self.cfg.robot_cfg)
        self._niro = RigidObject(self.cfg.niro_cfg)
        self._agv_contact = ContactSensor(self.cfg.agv_contact_cfg)
        self._niro_contact = ContactSensor(self.cfg.niro_contact_cfg)
        self._rcam = TiledCamera(self.cfg.rcam)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion and sensors to scene
        self.scene.articulations["agv"] = self._agv
        self.scene.rigid_objects["niro"] = self._niro
        self.scene.sensors["agv_contact"] = self._agv_contact
        self.scene.sensors["niro_contact"] = self._niro_contact
        self.scene.sensors["rcam"] = self._rcam

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.my_visualizer = define_markers()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        if torch.isnan(self.actions).any():
            raise ValueError("Actions contain NaN values.")
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._XY_PRI_idx + self._RR_RPIN_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._PY_PX_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._PZ_PY_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._PY_PX_PRI_idx)
        # self._agv.set_joint_effort_target(self.actions, joint_ids=self._RR_RPIN_PRI_idx)
        self._agv.set_joint_position_target(self.actions, joint_ids=self.actuated_dof_indices)

    def _get_observations(self) -> dict:
        self.calculate_values()

        data_type = "rgb"
        tensor = self._rcam.data.output[data_type].clone()[:, :, :, :3] 
        image = tensor.permute(0, 3, 1, 2).float() / 255.0
        # grayscale = transforms.Grayscale(1)
        # normalize = transforms.Normalize(0.0, 1.0)
        # grayscale_image = grayscale(image)
        # self.yolo_show.predict(image, show=True, verbose=False)
        results = self.yolo.predict(image, embed=[22], verbose=False, half=True)
        features = torch.stack(results)

        if not self.cfg.LSTM:
            self.serial_frames[:, :, :-1] = self.serial_frames[:, :, 1:].clone()
            self.serial_frames[:, :, -1] = features

        values = torch.cat(
            (
                math_utils.normalize(self._agv.data.joint_pos),
                math_utils.normalize(self._agv.data.joint_vel),
                math_utils.normalize(self._agv.data.joint_acc),
            ),
            dim=-1,
        )

        if self.cfg.write_image_to_file:
            array = image.cpu().numpy()
    
            for i in range(array.shape[0]):
                image_array = array[i]
                image_array = image_array.astype("uint8")

                # 이미지 생성 및 저장
                img = Image.fromarray(image_array)
                img.save("skrl_test/train_images/ff.png")
            self.save_image = False
            # self.image_counter += 1

        observations = {
            "policy": {
                "value": values,
                "image": features if self.cfg.LSTM else self.serial_frames,
                # "critic": torch.cat(
                #     (
                #         values,
                #         # math_utils.normalize(
                #         #     self._agv.data.body_state_w[
                #         #         :, self.actuated_dof_indices
                #         #     ].view(self.num_envs, self.num_actions * 13)
                #         # ),
                #         # self._niro.data.body_pos_w[:, 0] - self.scene.env_origins,
                #         self.pin_position(True) - self._agv.data.root_pos_w,
                #         self.pin_position(False) - self._agv.data.root_pos_w,
                #         self.hole_position(True) - self._agv.data.root_pos_w,
                #         self.hole_position(False) - self._agv.data.root_pos_w,
                #         self.pin_velocity(True),
                #         self.pin_velocity(False),
                #     ),
                #     dim=-1,
                # ),
            },
            # "critic": self._get_states(),
        }

        # print(observations["policy"]["critic"].shape)

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # reward
        direction = "r"
        rew_pin_r = self.pin_reward(True)
        rew_pin_r_xy = -(self.current_values[f"{direction}_xy_distance"] ** 2)
        r_z_penalty = self.current_values["r_z_distance"] ** 3

        pin_vel = self.current_values[f"{direction}_pin_vel"] + 1e-8
        rew_pin_vel = pin_vel ** 2

        correct_xy_rew = (
            self.current_values[f"{direction}_pin_correct_xy"].int()
            * torch.clamp(
                1 / pin_vel,
                max=10000,
                min=10
            )
        )
        # correct_z_rew = self.current_values[f"{direction}_pin_correct_z"].int() * 10
        correct_rew = (
            self.current_values[f"{direction}_pin_correct"].int()
            * torch.clamp(
                1 / rew_pin_vel,
                max=100000,
                min=100
            )
        )

        # penalty
        z_penalty = self.current_values["terminate_z"].int()
        contact_penalty = is_undesired_contacts(self._niro_contact).int()
        torque_penalty = torch.sum(self.current_values["agv_torque"] ** 2, dim=1)
        # sum
        total_reward = (
            rew_pin_r * self.reward_weights["rew_pin_r"]
            + rew_pin_r_xy * self.reward_weights["rew_pin_r_xy"]
            + correct_rew * self.reward_weights["correct_rew"]
            + rew_pin_vel * self.reward_weights["rew_pin_vel"]
            + z_penalty * self.reward_weights["z_penalty"]
            + contact_penalty * self.reward_weights["contact_penalty"]
            + torque_penalty * self.reward_weights["torque_penalty"]
            + correct_xy_rew * self.reward_weights["correct_xy_rew"]
            + r_z_penalty * self.reward_weights["r_z_penalty"]
            # + 1
            # + self.episode_length_buf * 0.1
            # + correct_z_rew
        )

        UP = "\x1b[3A"
        print(
            f"\npin: {round(rew_pin_r[0].item() * self.reward_weights['rew_pin_r'], 3)} "
            f"co: {self.current_values[f'{direction}_pin_correct'][0].item()} "
            f"z: {round(z_penalty[0].item() * self.reward_weights['z_penalty'], 2)} "
            f"vel: {round(rew_pin_vel[0].item() * self.reward_weights['rew_pin_vel'], 3)} "
            f"torque: {round(torque_penalty[0].item() * self.reward_weights['torque_penalty'], 2)} "
            f"contact: {round(contact_penalty[0].item() * self.reward_weights['contact_penalty'], 2)} "
            f"total: {round(total_reward[0].item(), 2)}_\n{UP}\r"
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # print(9)
        # self.joint_pos = self._agv.data.root_pos_w
        # self.joint_vel = self._agv.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pin_out_of_hole = self.current_values["terminate_z"]
        pin_in_hole = self.current_values["r_pin_correct"]
        pin_vel = self.current_values["r_pin_vel"] + 1e-8 < 0.01
        pin_correct = torch.logical_and(pin_in_hole, pin_vel)
        # torch.logical_or(pin_correct, pin_out_of_hole)
        if pin_correct.any():
            print(f"pin_correct: {torch.nonzero(pin_vel, as_tuple=True)[0]}")
        return pin_correct, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        if self.cfg.curriculum_learning:
            self.curriculum_switch()

        self.serial_frames = torch.zeros(
            self.serial_frames.shape, 
            dtype=torch.float, 
            device=self.device
        )
        
        self.randomize_joints_by_offset(
            env_ids,
            (-0.03, 0.03)
            if not self.cfg.curriculum_learning or self.random_pin_position
            else (0, 0),
            "agv"
        )
        
        self.randomize_object_position(
            env_ids,
            (-0.1, 0.1) if not self.cfg.curriculum_learning or self.random_hole_position else (0, 0),
            (-0.03, 0.03) if not self.cfg.curriculum_learning or self.random_hole_position else (0, 0), 
            "niro"
        )

        # set joint positions with some noise
        # joint_pos, joint_vel = self._agv.data.default_joint_pos.clone(), self._agv.data.default_joint_vel.clone()
        # joint_pos += torch.rand_like(joint_pos) * 0.1
        # self._agv.write_joint_state_to_sim(joint_pos, joint_vel)
        # clear internal buffers
        # self._agv.reset()

        object_names = ["AGV", "Niro"]
        material_names = ["OmniSurfaceLite", "material_silver"]
        property_names = [
            "Shader.inputs:diffuse_reflection_color",
            "Shader.inputs:diffuse_color_constant",
        ]
        stage = stage_utils.get_current_stage()

        for idx, object_name in enumerate(object_names):
            for env_id in env_ids:
                if not self.cfg.curriculum_learning or self.random_color:
                    color = Gf.Vec3f(random.random(), random.random(), random.random())
                    color_spec = stage.GetAttributeAtPath(f"/World/envs/env_{env_id}/{object_name}/Looks/{material_names[idx]}/{property_names[idx]}")
                    color_spec.Set(color)
                self.initial_pin_position(env_id)

    """
    custom functions
    """

    def reset_scene_to_default(self, env_ids: torch.Tensor):
        """Reset the scene to the default state specified in the scene configuration."""
        # rigid bodies
        for rigid_object in self.scene.rigid_objects.values():
            # obtain default and deal with the offset for env origins
            default_root_state = rigid_object.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            # set into the physics simulation
            rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)
        # articulations
        for articulation_asset in self.scene.articulations.values():
            # obtain default and deal with the offset for env origins
            default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            # set into the physics simulation
            articulation_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)
            # obtain default joint positions
            default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
            default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
            # set into the physics simulation
            articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    def randomize_joints_by_offset(
        self,
        env_ids: torch.Tensor,
        position_range: tuple[float, float],
        joint_name: str,
        # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        asset: Articulation = self.scene.articulations[joint_name]
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_pos += math_utils.sample_uniform(
            *position_range,
            joint_pos.shape,
            joint_pos.device,
        )
        joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        asset.write_joint_state_to_sim(joint_pos, 0, env_ids=env_ids)

    def randomize_object_position(
        self,
        env_ids: torch.Tensor,
        xy_position_range: tuple[float, float],
        z_position_range: tuple[float, float],
        joint_name: str,
        # asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        rigid_object = self.scene.rigid_objects[joint_name]
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += self.scene.env_origins[env_ids]

        xy_low, xy_high = xy_position_range
        z_low, z_high = z_position_range

        # Random offsets for X and Y coordinates
        xy_random_offsets = torch.tensor(
            np.random.uniform(xy_low, xy_high, size=(default_root_state.shape[0], 2)),  # For X and Y only
            dtype=default_root_state.dtype,
            device=default_root_state.device,
        )

        # Random offsets for Z coordinate
        z_random_offsets = torch.tensor(
            np.random.uniform(z_low, z_high, size=(default_root_state.shape[0], 1)),  # For Z only
            dtype=default_root_state.dtype,
            device=default_root_state.device,
        )

        # Apply random offsets to the X, Y, and Z coordinates
        default_root_state[:, 0:2] += xy_random_offsets  # Apply to X and Y coordinates
        default_root_state[:, 2:3] += z_random_offsets  # Apply to Z coordinate

        # set into the physics simulation
        rigid_object.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    def calculate_values(self):
        r_pin_pos = self.pin_position(True)
        l_pin_pos = self.pin_position(False)
        r_pin_vel = self.pin_velocity(True)
        l_pin_vel = self.pin_velocity(False)
        r_hole_pos = self.hole_position(True)
        l_hole_pos = self.hole_position(False)

        r_distance = euclidean_distance(r_hole_pos, r_pin_pos)
        l_distance = euclidean_distance(l_hole_pos, l_pin_pos)
        r_xy_distance = euclidean_distance(r_hole_pos[:,:2], r_pin_pos[:,:2])
        l_xy_distance = euclidean_distance(l_hole_pos[:,:2], l_pin_pos[:,:2])
        r_z_distance = r_hole_pos[:,2] - r_pin_pos[:,2]
        l_z_distance = l_hole_pos[:,2] - l_pin_pos[:,2]

        agv_torque = self._agv.data.applied_torque[:,self.actuated_dof_indices]
        r_pin_lv = r_pin_vel[..., :3]
        r_pin_v_norm = torch.norm(r_pin_lv, dim=-1)
        l_pin_lv = l_pin_vel[..., :3]
        l_pin_v_norm = torch.norm(l_pin_lv, dim=-1)

        self.current_values = dict(
            agv_torque=agv_torque,
            r_pin_pos=r_pin_pos,
            l_pin_pos=l_pin_pos,
            r_pin_vel=r_pin_v_norm,
            l_pin_vel=l_pin_v_norm,
            r_hole_pos=r_hole_pos,
            l_hole_pos=l_hole_pos,
            r_distance=r_distance,
            l_distance=l_distance,
            r_xy_distance=r_xy_distance,
            l_xy_distance=l_xy_distance,
            r_z_distance=r_z_distance,
            l_z_distance=l_z_distance,
            r_pin_correct=r_distance < 0.01,
            l_pin_correct=l_distance < 0.01,
            r_pin_correct_xy=r_xy_distance < 0.01,
            l_pin_correct_xy=l_xy_distance < 0.01,
            r_pin_correct_z=r_z_distance < 0.01,
            l_pin_correct_z=l_z_distance < 0.01,
            terminate_z=torch.logical_and(r_pin_pos[:, 2] >= r_hole_pos[:, 2] + 0.03, r_xy_distance >= 0.01),
        )

        pin_list = [r_pin_pos[i].tolist() for i in range(self.num_envs)]
        hole_list = [r_hole_pos[i].tolist() for i in range(self.num_envs)]
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        draw.draw_lines(pin_list, hole_list, [(1, 1, 1, 1)] * self.num_envs, [5] * self.num_envs)

    def pin_position(self, right: bool = True, env_id = None):
        root_position: torch.Tensor = (
            self._agv.data.body_pos_w[env_id, self._RPIN_idx[0] if right else self._LPIN_idx[0], :]
            if env_id is not None
            else self._agv.data.body_pos_w[:, self._RPIN_idx[0] if right else self._LPIN_idx[0], :]
        )
        pin_rel = torch.tensor(
            [0, 0.02 if right else -0.02, 0.479],
            device="cuda:0",
        )  # 0.479
        pin_pos_w = root_position + pin_rel
        return pin_pos_w
    
    def pin_velocity(self, right: bool = True, env_id = None):
        pin_vel_w: torch.Tensor = (
            self._agv.data.body_vel_w[env_id, self._RPIN_idx[0] if right else self._LPIN_idx[0], :]
            if env_id is not None
            else self._agv.data.body_vel_w[:, self._RPIN_idx[0] if right else self._LPIN_idx[0], :]
        )
        return pin_vel_w

    def hole_position(self, right: bool = True, env_id = None):
        # niro: RigidObject = self.scene.rigid_objects["niro"]
        niro_pos = (
            self._niro.data.root_pos_w[env_id]  # torch.tensor([-0.5000,  0.0000,  1.1000], device="cuda:0")
            if env_id is not None
            else self._niro.data.root_pos_w
        )
        hole_rel = torch.tensor(
            [0.455, 0.693 if right else -0.693, 0.0654 - 0.03],
            device="cuda:0",
        )
        hole_pos_w = torch.add(niro_pos, hole_rel)

        # l hole [.455, .693, .0654] 33.5 (1)/ 50.8(2) / 65.4(3) / 75.5(all)
        # niro [-0.5000,  0.0000,  1.1000]
        # lpin [0, .163, .514 / .479]
        return hole_pos_w

    def pin_reward(self, right: bool = True) -> torch.Tensor:
        """
        [보상]
        XYZ 좌표의 일치: 처음 거리 * w

        XY 좌표의 일치: 처음 XY거리 * w
        Z 좌표의 일치: 처음 Z거리 * w

        [변수]

        전 프레임 대비 상대적 XY 좌표의 변화: (prev_xy_distance - curr_xy_distance) * w
        전 프레임 대비 상대적 Z 좌표의 변화: (prev_z_distance - curr_z_distance) * w

        [패널티]
        niro contact
        Z 좌표 실패
        처음 위치에서 벗어나지 않으면 episode 비례 패널티

        [초기화]
        z좌표 실패
        XYZ 일치
        Out-of-bounds?

        [Randomize]
        niro color
        pin color
        """

        direction = "r" if right else "l"

        hole_pos_w = self.current_values[f"{direction}_hole_pos"]
        curr_pin_pos_w = self.current_values[f"{direction}_pin_pos"]
        prev_pin_pos_w = self.prev_pos_w[f"{direction}_pin"]

        # prev_pin_xy = prev_pin_pos_w[:, 0:1]
        # prev_xy_distance = euclidean_distance(hole_pos_w[:, 0:1], prev_pin_xy)
        # relative_xy_rew = prev_xy_distance - curr_xy_distance

        # curr_z_dist = self.current_values[f"{direction}_z_distance"]
        # curr_z_rew = self.init_z_distance_r - curr_z_dist

        # prev_pin_z = prev_pin_pos_w[:, 2]
        # prev_z_dist = hole_pos_w[:, 2] - prev_pin_z
        # relative_z_rew = prev_z_dist - curr_z_dist

        dist1 = euclidean_distance(self.init_pin_pos, curr_pin_pos_w)
        dist2 = self.current_values[f"{direction}_distance"]
        dist3 = dist1 + dist2 - self.init_distance_r
        rew = dist3 ** 3

        self.prev_pos_w[f"{direction}_pin"] = curr_pin_pos_w

        curr_xyz_dist = self.current_values[f"{direction}_distance"]
        xyz_rew = (self.init_distance_r - curr_xyz_dist) ** 3

        # prev_xyz_dist = euclidean_distance(hole_pos_w, prev_pin_pos_w)
        # relative_xyz_dist = prev_xyz_dist - curr_xyz_dist
        # relative_xyz_rew = relative_xyz_dist ** 3

        reward = xyz_rew - rew

        # UP = "\x1b[3A"
        # print(  # xy: {curr_xy_rew[0]} z: {curr_z_rew[0]} rxy: {round(relative_xy_rew[0].item(), 3)} rz: {round(relative_z_rew[0].item(), 3)}
        #     f"\nrew: {round(reward[0].item(), 4)}_\n{UP}\r"
        # )

        return reward

    def pin_direction_reward(self, hole, pin) -> torch.Tensor:
        current_direction = pin - hole
        current_direction = current_direction / torch.norm(current_direction, p=2)
        cosine_similarity = torch.dot(current_direction, self.init_direction)
        return cosine_similarity

    def initial_pin_position(self, env_id: int):
        r_pin_pos_w = self.pin_position(True, env_id)
        l_pin_pos_w = self.pin_position(False, env_id)
        l_hole_pos_w = self.hole_position(False, env_id)
        r_hole_pos_w = self.hole_position(True, env_id)

        self.init_distance_l[env_id] = euclidean_distance(l_pin_pos_w, l_hole_pos_w)
        self.init_distance_r[env_id] = euclidean_distance(r_pin_pos_w, r_hole_pos_w)
        self.init_pin_pos[env_id] = r_pin_pos_w
        self.init_hole_pos[env_id] = r_hole_pos_w
        self.init_direction[env_id] = r_hole_pos_w - r_pin_pos_w

        r_hole_z = r_hole_pos_w[2]
        r_pin_z = r_pin_pos_w[2]

        self.init_z_distance_r[env_id] = r_hole_z - r_pin_z

        r_hole_xy = r_hole_pos_w[0:1]
        r_pin_xy = r_pin_pos_w[0:1]

        self.init_xy_distance_r[env_id] = euclidean_distance(r_hole_xy, r_pin_xy)

        self.prev_pos_w["r_pin"][env_id] = r_pin_pos_w
        self.prev_pos_w["l_pin"][env_id] = l_pin_pos_w

        # marker_locations = torch.vstack(
        #     (
        #         self.init_hole_pos,
        #         self.init_pin_pos# - torch.tensor([0, 0, 0.479], device="cuda:0"),
        #     )
        # )
        # self.my_visualizer.visualize(marker_locations)

    def curriculum_switch(self):
        multiplier = 100

        if self.common_step_counter < 1000 * multiplier:
            self.reward_weights = {
                "rew_pin_r": 0,
                "rew_pin_r_xy": 100,
                "correct_xy_rew": 0,
                "correct_rew": 0,
                "z_penalty": -100,
                "contact_penalty": 0,
                "torque_penalty": 0,
                "r_z_penalty": 100,
            }
        elif 2000 * multiplier < self.common_step_counter < 3000 * multiplier:
            self.random_color = True
            self.reward_weights["torque_penalty"] = -0.00003
        elif 3000 * multiplier < self.common_step_counter < 4000 * multiplier:
            self.random_hole_position = True
            self.reward_weights = {
                "rew_pin_r": 1000,
                "rew_pin_r_xy": 0,
                "correct_xy_rew": 0,
                "correct_rew": 0,
                "z_penalty": -100,
                "contact_penalty": 0,
                "torque_penalty": 0,
                "r_z_penalty": -100,
            }
        elif 4000 * multiplier < self.common_step_counter < 5000 * multiplier:
            self.random_color = True
            self.reward_weights["torque_penalty"] = -0.00003
        elif 5000 * multiplier < self.common_step_counter < 6000 * multiplier:
            self.random_hole_position = True
            self.reward_weights = {
                "rew_pin_r": 1000,
                "rew_pin_r_xy": 0,
                "correct_xy_rew": 0,
                "correct_rew": 1,
                "z_penalty": -100,
                "contact_penalty": 0,
                "torque_penalty": 0,
                "r_z_penalty": 0,
            }
        elif 6000 * multiplier < self.common_step_counter < 6000 * multiplier:
            self.random_pin_position = True
            self.reward_weights["contact_penalty"] = -.1
        elif 7000 * multiplier < self.common_step_counter < 8000 * multiplier:
            self.reward_weights["torque_penalty"] = -0.00003


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def euclidean_distance(src, dist):
    distance = torch.sqrt(torch.sum((src - dist) ** 2, dim=src.ndim-1) + 1e-8)
    return distance

def power_reward(reward) -> torch.Tensor:
    r = torch.where(reward < 0, -((reward - 1) ** 2), (reward + 1) ** 2)
    return r

def is_undesired_contacts(sensor: ContactSensor) -> torch.Tensor:
    net_contact_forces: torch.Tensor = sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, 0], dim=-1), dim=1)[0] > 0
    return is_contact


def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
    world_transform = xformable.ComputeLocalToWorldTransform(0)
    world_pos = world_transform.ExtractTranslation()
    world_quat = world_transform.ExtractRotationQuat()

    px = world_pos[0] - env_pos[0]
    py = world_pos[1] - env_pos[1]
    pz = world_pos[2] - env_pos[2]
    qx = world_quat.imaginary[0]
    qy = world_quat.imaginary[1]
    qz = world_quat.imaginary[2]
    qw = world_quat.real

    return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)