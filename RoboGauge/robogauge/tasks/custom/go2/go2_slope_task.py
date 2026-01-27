from robogauge.tasks.gauge import SlopeForwardGaugeConfig, SlopeBackwardGaugeConfig
from robogauge.tasks.simulator.mujoco_config import MujocoConfig

class Go2SlopeForwardGaugeConfig(SlopeForwardGaugeConfig):
    class metrics(SlopeForwardGaugeConfig.metrics):
        class dof_limits(SlopeForwardGaugeConfig.metrics.dof_limits):
            enabled = True
            soft_dof_limit_ratio = 0.4
            dof_names = ['hip', 'thigh']  # List of DOF names to monitor, None for all

class Go2SlopeBackwardGaugeConfig(SlopeBackwardGaugeConfig):
    class metrics(SlopeBackwardGaugeConfig.metrics):
        class dof_limits(SlopeBackwardGaugeConfig.metrics.dof_limits):
            enabled = True
            soft_dof_limit_ratio = 0.4
            dof_names = ['hip', 'thigh']  # List of DOF names to monitor, None for all
