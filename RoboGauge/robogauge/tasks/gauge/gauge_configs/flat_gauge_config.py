from robogauge.tasks.gauge.base_gauge_config import BaseGaugeConfig

class FlatGaugeConfig(BaseGaugeConfig):
    gauge_class = 'BaseGauge'

    class assets(BaseGaugeConfig.assets):
        terrain_name = "flat"
        terrain_level = 0
        terrain_xmls = ['{ROBOGAUGE_ROOT_DIR}/resources/terrains/flat.xml']
        terrain_spawn_pos = [0, 0, 0]  # x y z [m], robot freejoint spawn position on the terrain
