from robogauge.tasks.gauge import ObstacleGaugeConfig
from robogauge.tasks.simulator.mujoco_config import MujocoConfig

class Go2ObstacleGaugeConfig(ObstacleGaugeConfig):
    class metrics(ObstacleGaugeConfig.metrics):
        class dof_limits(ObstacleGaugeConfig.metrics.dof_limits):
            enabled = True
            soft_dof_limit_ratio = 0.4
            dof_names = ['hip', 'thigh']  # List of DOF names to monitor, None for all
