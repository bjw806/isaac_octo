import gymnasium as gym


gym.register(
    id="Isaac-AGV-Direct",
    entry_point=f"{__name__}.direct_agv_env:AGVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_agv_env:AGVEnvCfg",
    },
)