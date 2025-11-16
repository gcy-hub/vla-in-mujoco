from env.base_env import CustomViewer
import mujoco
import numpy as np
import mujoco.viewer
import glfw
import cv2


class MultiViewer(CustomViewer):
    def __init__(self, model_path):
        super(MultiViewer, self).__init__(model_path)
        # 设置相机
        self.camera_names = ["wrist_camera", "top_camera"]
        self.cameras = []

        # 随机初始化方块参数
        self.table_top_z = 0.75
        self.cube_half_size = 0.02  # x,y 半边长

        # 根据机械臂的工作空间范围来确定
        self.x_min = 0.2
        self.x_max = 0.3
        self.y_min = -0.2
        self.y_max = 0.2

        # 对应 xml 里三个方块的 body 名称
        self.cube_bodies = ["cube_red", "cube_green", "cube_blue"]

        # 离屏渲染
        self.resolution = (640, 480)
        self.init_glfw()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # 初始化相机
        self.init_camera()

        # 创建帧缓冲对象
        self.framebuffer = mujoco.MjrRect(0, 0, self.resolution[0], self.resolution[1])
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        # ---------- 初始化时随机方块位置 ----------
        # self.init_cubes_random()
        mujoco.mj_forward(self.model, self.data)

        # 用来检测“是否被 Reset”
        self.prev_time = self.data.time

    def init_glfw(self):
        # 创建OpenGL上下文（离屏渲染）
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(self.resolution[0], self.resolution[1], "Offscreen", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)

    def init_camera(self):
        for camera_name in self.camera_names:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            if camera_id != -1:
                print("camera_id", camera_id)
                camera.fixedcamid = camera_id
            self.cameras.append(camera)

    def init_cubes_random(self):
        """
        随机初始化三个方块在桌面上的位置（x, y），z 放在桌面上方，
        orient 和线速度/角速度清零。
        """
        for body_name in self.cube_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
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
            self.data.qvel[qvel_adr: qvel_adr + 6] = 0.0

    def on_reset(self):
        """
        检测到 MuJoCo 被 Reset 后调用：
        不用 reset_callback，纯在 Python 侧覆盖一下状态。
        """
        if self.data.time < self.prev_time - 1e-12:
            # 不再调用 mj_resetData，因为 GUI 那边已经 reset 过了
            # self.init_cubes_random()
            mujoco.mj_forward(self.model, self.data)


    def off_screen_render(self, viewports: list, cameras: list, window_names: list):
        assert len(viewports) == len(cameras) == len(window_names), \
            f"{len(viewports), len(cameras), len(window_names)} the length of viewport camera window_name should be the same"
        for index in range(len(viewports)):
            mujoco.mjv_updateScene(
                self.model, self.data,
                mujoco.MjvOption(),
                mujoco.MjvPerturb(),
                cameras[index],
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene
            )
            mujoco.mjr_render(viewports[index], self.scene, self.context)
            rgb = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, viewports[index], self.context)
            # 转换颜色空间 (OpenCV使用BGR格式)
            bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_names[index], bgr)

    def render(self):
        # 同步 prev_time
        self.on_reset()
        self.prev_time = self.data.time
        # 创建多视角
        viewport1 = mujoco.MjrRect(0, 0, self.resolution[0], self.resolution[1])
        viewport2 = mujoco.MjrRect(0, 0, self.resolution[0], self.resolution[1])
        # 离屏渲染
        self.off_screen_render([viewport1, viewport2], self.cameras, self.camera_names)
        cv2.waitKey(1)

    def runFunc(self):
        self.render()

    def runAfter(self):
        del self.context
        del self.scene
        cv2.destroyAllWindows()
        glfw.terminate()