from env.multi_camera_env import MultiViewer
import numpy as np
from lerobot.teleoperators import (
    make_teleoperator_from_config,
    so101_leader,
)

class MultiViewerWithLeader(MultiViewer):
    def __init__(self, model_path, teleop_id='my_awesome_leader_arm', teleop_port='/dev/ttyACM1'):
        super(MultiViewerWithLeader, self).__init__(model_path)
        # 初始化遥控机械臂
        self.teleop_cfg = so101_leader.SO101LeaderConfig(id=teleop_id, calibration_dir=None,
                                                    port=teleop_port, use_degrees=False)
        self.teleop = make_teleoperator_from_config(self.teleop_cfg)
        self.teleop.connect()

        # 定义六关节的范围（单位：弧度）
        self.target_ranges = [
            (-1.92, 1.92),
            (-3.32, 0.174),
            (-0.174, 3.14),
            (-1.66, 1.66),
            (-2.79, 2.79),
            (-0.174, 1.75)
        ]
        self.current_ranges = [
            (-1.5751544616690798, 1.6152769732091794),
            (-1.6781753754266369, 1.7224969339613139),
            (-1.7326876639309934, 1.7263668698993255),
            (-1.7143208642685421, 1.7113676844851335),
            (-1.735243293616637, 1.737994009537826),
            (0.027017480681026777, 1.6723820541555574)
        ]

        print("teleop connect on port {teleop_port}".format(teleop_port=teleop_port))

    def map_range(self, x: float, current_range, target_range) -> float:
        """ 数值范围转换 """
        x_min, x_max, y_min, y_max = current_range[0], current_range[1], target_range[0], target_range[1]

        if x_max == x_min:
            raise ValueError(f"x_max ({x_max}) and x_min ({x_min}) cannot be the same")

        y = y_min + (x - x_min) * (y_max - y_min) / (x_max - x_min)
        return y

    def get_teleop_action(self):
        raw_action = self.teleop.get_action()

        action = [
            -self.map_range(np.radians(raw_action['shoulder_pan.pos']), self.current_ranges[0], self.target_ranges[0]),
            self.map_range(np.radians(raw_action['shoulder_lift.pos']), self.current_ranges[1], self.target_ranges[1]),
            self.map_range(np.radians(raw_action['elbow_flex.pos']), self.current_ranges[2], self.target_ranges[2]),
            self.map_range(np.radians(raw_action['wrist_flex.pos']), self.current_ranges[3], self.target_ranges[3]),
            self.map_range(np.radians(raw_action['wrist_roll.pos']), self.current_ranges[4], self.target_ranges[4]),
            self.map_range(np.radians(raw_action['gripper.pos']), self.current_ranges[5], self.target_ranges[5]),
        ]

        return action

    # 获取真实机械臂关节角度
    def runFunc(self):
        self.render()
        action = self.get_teleop_action()
        self.data.qpos[:6] = action
        print(f"{action=}")


