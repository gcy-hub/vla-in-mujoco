import mujoco
import numpy as np
import cv2
import glfw
import mujoco.viewer
from typing import Dict, List, Tuple


class SimRobot:
    """
    仿真中的机械臂 + 场景封装。

    功能：
    1. 利用 MuJoCo xml 初始化机械臂与场景；
    2. 提供观测：六个关节角度 + 两个相机图像；
    3. 提供更新关节动作接口（直接写 qpos）；
    4. 管理离屏渲染和随机方块场景。
    """

    def __init__(
        self,
        model_path: str,
        resolution: Tuple[int, int] = (640, 480),
        distance: float = 2.3,
        azimuth: float = 45.0,
        elevation: float = -60.0,
        lookat: List[float] = None,
    ) -> None:
        if lookat is None:
            lookat = [0.4, 0.0, 0.0]

        self.resolution = resolution

        # 观测中的相机名称映射：键是输出的 key，值是 xml 中的 camera 名
        self.camera_names: Dict[str, str] = {
            "front": "top_camera",
            "side": "wrist_camera",
        }

        # --------- 初始化仿真、viewer、摄像头与场景 ---------
        self._init_mujoco_simulation(model_path, distance, azimuth, elevation, lookat)
        self._init_offscreen_rendering()
        self._init_cameras()

        # 初始化机械臂关节名称（与 get_observation / update_actions 对齐）
        self.joint_names: List[str] = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        # 初始化机械臂关节角 + 随机场景
        self._init_scene_objects()

        print("场景创建完成")

    def _init_mujoco_simulation(
        self,
        model_path: str,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: List[float],
    ) -> None:
        """加载 MuJoCo 模型并配置主 viewer 视角。"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 主界面被动 viewer
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)

        # 设置主视角摄像头
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation
        self.handle.cam.lookat = lookat
        
        # xml文件中关节的最大最小范围
        self.target_ranges = [
            (-1.92, 1.92),
            (-3.32, 0.174),
            (-0.174, 3.14),
            (-1.66, 1.66),
            (-2.79, 2.79),
            (-0.174, 1.75)
        ]
        # so-arm101主臂输出的范围
        self.current_ranges = [
            (-1.5751544616690798, 1.6152769732091794),
            (-1.6781753754266369, 1.7224969339613139),
            (-1.7326876639309934, 1.7263668698993255),
            (-1.7143208642685421, 1.7113676844851335),
            (-1.735243293616637, 1.737994009537826),
            (0.027017480681026777, 1.6723820541555574)
        ]

    def _init_offscreen_rendering(self) -> None:
        """初始化离屏渲染（GLFW 窗口、Scene、Context、Viewport）。"""
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(
            self.resolution[0],
            self.resolution[1],
            "Offscreen",
            None,
            None,
        )
        glfw.make_context_current(self.window)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
        )
        self.viewport = mujoco.MjrRect(
            0,
            0,
            self.resolution[0],
            self.resolution[1],
        )

        self._set_buffer()

    def _init_cameras(self) -> None:
        """初始化用于离屏渲染的相机对象。"""
        self.top_camera = mujoco.MjvCamera()
        self.wrist_camera = mujoco.MjvCamera()

    def _init_scene_objects(self) -> None:
        """初始化机械臂姿态与随机方块场景。"""
        # 初始化机械臂关节
        self.init_joint()

        # 随机初始化场景中的方块
        self.randomScene = Scene(self.model, self.data)
        # self.randomScene.init_cubes_random()
        mujoco.mj_forward(self.model, self.data)

    def init_joint(self) -> None:
        """设置机械臂初始关节角。"""
        self.joint_init = [0, -1.75, 1.69, 0.663, 0, 0]
        self.data.qpos[:6] = self.joint_init
        self.data.ctrl[:6] = self.joint_init
        mujoco.mj_forward(self.model, self.data)

    def reset_sim(self, manual: bool = False) -> None:
        """
        检测到 MuJoCo 被 Reset 后重置场景。

        参数:
            manual: 外部手动请求重置（True 时无视时间判断，强制重置）。
        """
        # 检测到 mujoco reset 按键点击或者手动复位
        if self.data.time < self.randomScene.prev_time - 1e-12 or manual:
            # 不再调用 mj_resetData，因为 GUI 那边已经 reset 过了
            print("reset sim")
            # 使用 MuJoCo 原生 resetData，重置 qpos,qvel 等为模型默认状态
            mujoco.mj_resetData(self.model, self.data)  # 重置状态到模型 qpos0 等默认值 :contentReference[oaicite:1]{index=1}
            # self.randomScene.init_cubes_random()
            self.init_joint()
            mujoco.mj_forward(self.model, self.data)

        self.randomScene.prev_time = self.data.time

    def _set_buffer(self) -> None:
        """将渲染目标设置为离屏 framebuffer。"""
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        返回虚拟环境中的观测空间，包括：
        - 机械臂的自身状态（六个关节角度）；
        - 两个相机图像（front / side）。

        返回:
            obs: Dict {
                'shoulder_pan.pos', ...,
                'gripper.pos',
                'front',  # 顶视图像
                'side',   # 手腕视图像
            }
        """
        obs: Dict[str, np.ndarray] = {}

        # 关节状态（保持原实现：从 qpos[:6] 直接读取）
        for index, joint_name in enumerate(self.joint_names):
            obs[joint_name] = self.data.qpos[index]

        # 相机图像
        front_image = self._render_camera_image(
            self.top_camera, self.camera_names["front"]
        )
        side_image = self._render_camera_image(
            self.wrist_camera, self.camera_names["side"]
        )
        obs["front"] = front_image
        obs["side"] = side_image

        return obs

    def update_actions(self, actions: np.ndarray) -> None:
        """将动作（关节目标值）写入 qpos，并进行前向计算。"""
        self.data.qpos[:6] = actions
        mujoco.mj_forward(self.model, self.data)

    def offscreen_render(self) -> None:
        """
        从两个相机渲染图像并用 OpenCV 显示。
        调试用，逻辑与原实现一致。
        """
        front_image = self._render_camera_image(
            self.top_camera, self.camera_names["front"]
        )
        side_image = self._render_camera_image(
            self.wrist_camera, self.camera_names["side"]
        )

        bgr_front = cv2.cvtColor(np.flipud(front_image), cv2.COLOR_RGB2BGR)
        bgr_side = cv2.cvtColor(np.flipud(side_image), cv2.COLOR_RGB2BGR)

        cv2.imshow("Frone Camera", bgr_front)  # 保持你原来的拼写
        cv2.imshow("Side Camera", bgr_side)
        cv2.waitKey(1)

    def exit_sim(self) -> None:
        """释放 OpenCV 和 GLFW 资源。"""
        cv2.destroyAllWindows()
        glfw.terminate()  # 清理资源

    def _render_camera_image(
        self,
        camera: mujoco.MjvCamera,
        camera_name: str,
    ) -> np.ndarray:
        """
        渲染指定名称的相机并返回 RGB 图像数组。

        参数:
            camera: MjvCamera 对象（top / wrist）
            camera_name: xml 中 camera 的 name

        返回:
            rgb_corrected: (H, W, 3) uint8, 已经垂直翻转的图像
        """
        # 根据相机名称获取相机ID
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if camera_id == -1:
            raise ValueError(f"未找到名为 '{camera_name}' 的相机")

        # 配置相机为固定类型，并关联到 XML 中定义的相机
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id

        # 更新场景（从相机的视角）
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            mujoco.MjvOption(),
            mujoco.MjvPerturb(),
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        # 渲染场景
        mujoco.mjr_render(self.viewport, self.scene, self.context)

        # 创建缓冲区并读取像素数据
        rgb = np.zeros(
            (self.viewport.height, self.viewport.width, 3),
            dtype=np.uint8,
        )
        depth = np.zeros(
            (self.viewport.height, self.viewport.width),
            dtype=np.float32,
        )

        mujoco.mjr_readPixels(rgb, depth, self.viewport, self.context)

        # 修正：垂直翻转图像（MuJoCo 原点在左下角，标准图像原点在左上角）
        rgb_corrected = np.flipud(rgb).copy()
        return rgb_corrected

    def map_range(self, x: float, current_range, target_range) -> float:
        """
        将原始数 x（假设 x_min <= x <= x_max）映射到目标范围 [y_min, y_max]。
        映射公式： y = y_min + (x - x_min) * (y_max - y_min) / (x_max - x_min)

        参数：
          x       -- 原始值
          x_min   -- 原始范围最小值
          x_max   -- 原始范围最大值
          y_min   -- 目标范围最小值
          y_max   -- 目标范围最大值

        返回：
          映射后的值（float）
        """
        x_min, x_max, y_min, y_max = current_range[0], current_range[1], target_range[0], target_range[1]
        # 防止除以零
        if x_max == x_min:
            raise ValueError(f"x_max ({x_max}) and x_min ({x_min}) cannot be the same")
        # 线性映射
        y = y_min + (x - x_min) * (y_max - y_min) / (x_max - x_min)
        return y

# 本来打算做抓取任务的，但是一直抓不住，之后再用上
class Scene:
    """
    管理桌面上三个方块的随机初始化。

    逻辑保持不变：
    - 在给定范围内随机采样 (x, y)，z 在桌面上；
    - 设置 freejoint 的位置与姿态；
    - 将线速度与角速度清零；
    - 用 prev_time 检测是否被 reset。
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        # 随机初始化方块参数（保持原来的数值）
        self.table_top_z = 0.75 - 0.05 + 0.02
        self.cube_half_size = 0.02  # x, y 半边长

        # 根据机械臂的工作空间范围来确定
        self.x_min = 0.2
        self.x_max = 0.3
        self.y_min = -0.1
        self.y_max = 0.1

        # 对应 xml 里三个方块的 body 名称
        self.cube_bodies = ["cube_red", "cube_green", "cube_blue"]

        # 用来检测“是否被 Reset”
        self.prev_time = self.data.time

    def init_cubes_random(self) -> None:
        """
        随机初始化三个方块在桌面上的位置 (x, y)，
        z 放在桌面上方，orient 和线速度/角速度清零。
        """
        for body_name in self.cube_bodies:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if body_id < 0:
                continue

            # body 的第一个 joint 的 qpos 索引（freejoint 有 7 个自由度）
            jnt_adr = self.model.body_jntadr[body_id]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]

            # 随机采样桌面上的 (x, y)
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            z = self.table_top_z + self.cube_half_size  # 刚好放在桌面上

            # 位置
            self.data.qpos[qpos_adr + 0] = x
            self.data.qpos[qpos_adr + 1] = y
            self.data.qpos[qpos_adr + 2] = z

            # 四元数设置为单位旋转 (w, x, y, z) = (1, 0, 0, 0)
            self.data.qpos[qpos_adr + 3] = 1.0
            self.data.qpos[qpos_adr + 4] = 0.0
            self.data.qpos[qpos_adr + 5] = 0.0
            self.data.qpos[qpos_adr + 6] = 0.0

            # 对应的速度也清零
            qvel_adr = self.model.jnt_dofadr[jnt_adr]
            # freejoint 有 6 dof: 3 线速度 + 3 角速度
            self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
