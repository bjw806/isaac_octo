from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.direct.skrl_test.agv_cfg import AGV_JOINT

from .manage_agv_env import AGVEnvCfg


@configclass
class IKAGVEnvCfg(AGVEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.actions.PRI = DifferentialInverseKinematicsActionCfg(
            asset_name="agv",
            joint_names=[
                AGV_JOINT.PZ_PY_PRI,
                AGV_JOINT.PY_PX_PRI,
                AGV_JOINT.RR_RPIN_PRI,
            ],
            body_name="mb_1",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
        )
