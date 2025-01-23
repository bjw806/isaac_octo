import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
import omni.isaac.lab.sim as sim_utils


LIFT_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sites/IsaacLab/model/station/lift_2.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "J_LRX": 0.15,
            "J_RRX": 0.15,
        },
    ),
    actuators={
        "drop": ImplicitActuatorCfg(
            joint_names_expr=["J_LFZ", "J_LRZ", "J_RFZ", "J_RRZ"],
            effort_limit=100000000,
            stiffness=10000000.0,
            damping=0.0,
        ),
        "foot": ImplicitActuatorCfg(
            joint_names_expr=[
                "J_LFF",
                "J_LFR",
                "J_LRF",
                "J_LRR",
                "J_RFF",
                "J_RFR",
                "J_RRF",
                "J_RRR",
            ],
            stiffness=10000000.0,
            damping=100.0,
        ),
        "holder": ImplicitActuatorCfg(
            joint_names_expr=["J_LFY", "J_LRY", "J_RFY", "J_RRY"],
            stiffness=10000000.0,
            damping=0.0,
        ),
        "adjust": ImplicitActuatorCfg(
            joint_names_expr=["J_LRX", "J_RRX"],
            stiffness=10000000.0,
            damping=100000.0,
        ),
    },
)
