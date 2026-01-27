from robogauge.tasks.gauge import WaveGaugeConfig
from robogauge.tasks.simulator.mujoco_config import MujocoConfig

class Go2WaveGaugeConfig(WaveGaugeConfig):
    class metrics(WaveGaugeConfig.metrics):
        class dof_limits(WaveGaugeConfig.metrics.dof_limits):
            enabled = True
            soft_dof_limit_ratio = 0.4
            dof_names = ['hip', 'thigh']  # List of DOF names to monitor, None for all
