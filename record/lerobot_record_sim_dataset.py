
from record.config import RecordConfig
import logging
import time
import mujoco
import mujoco.viewer
from dataclasses import asdict
from pprint import pformat
from typing import Any
import numpy as np
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# 加载仿真中的机械臂场景
from robot import SimRobot


""" --------------- record_loop() data flow --------------------------
       [ SimRobot ]
           V
     [ sim_robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""

@safe_stop_image_writer
def record_loop(
        sim_robot,
        robot: Robot,
        events: dict,
        fps: int,
        teleop_action_processor: RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ],  # runs after teleop
        robot_action_processor: RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ],  # runs before robot
        robot_observation_processor: RobotProcessorPipeline[
            RobotObservation, RobotObservation
        ],  # runs after robot
        dataset: LeRobotDataset | None = None,
        teleop: Teleoperator | list[Teleoperator] | None = None,
        policy: PreTrainedPolicy | None = None,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
        control_time_s: int | None = None,
        single_task: str | None = None,
        display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s and sim_robot.handle.is_running():
        start_loop_t = time.perf_counter()

        # 提前结束标志挂起
        if events["exit_early"]:
            # 重启仿真
            sim_robot.reset_sim(manual=True)
            events["exit_early"] = False  # 关闭提前结束标志
            print('exit early and reset sim')
            break

        # 开启仿真
        mujoco.mj_forward(sim_robot.model, sim_robot.data)  # 自动前向计算
        # 离屏渲染
        # sim_robot.offscreen_render()
        # 检测mujoco界面中的reset按键
        sim_robot.reset_sim()

        # Get robot observation
        # obs = robot.get_observation()  # 这里的obs是关节角度
        # 这里用仿真环境来替换
        obs = sim_robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)  # 这里出来的值没有发生变化，留着

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            # 这两build完之后，变成了列表 ['observation.state', 'observation.images.front', 'observation.images.side']

        # Get action from either policy or teleop
        # 这里就相当于使用训练好的策略
        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )

            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)
        # 遥操作
        elif policy is None and isinstance(teleop, Teleoperator):
            # 这里的act就是so-arm101的六个电机关节角度(角度)
            act = teleop.get_action()
            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
        else:
            action_values = act_processed_teleop

        send_action = []
        for index, joint_name in enumerate(sim_robot.joint_names):
            # 如果这里使用的是so arm 101
            # send_action.append(np.radians(action_values[joint_name]))
            # 这里用的是so arm 100
            if index == 0:
                send_action.append(
                    -sim_robot.map_range(np.radians(action_values[joint_name]), sim_robot.current_ranges[index],
                                         sim_robot.target_ranges[index]))
            else:
                send_action.append(
                    sim_robot.map_range(np.radians(action_values[joint_name]), sim_robot.current_ranges[index],
                                        sim_robot.target_ranges[index]))

        # 将得到的动作序列发送给仿真机械臂
        sim_robot.update_actions(send_action)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t

        mujoco.mj_step(sim_robot.model, sim_robot.data)
        sim_robot.handle.sync()


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 构建仿真环境
    sim_robot = SimRobot(cfg.model_path)

    # 可视化数据
    if cfg.display_data:
        init_rerun(session_name="recording")

    # 初始化硬件设备: robot 从臂 teleop 主臂
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # 数据处理
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features), # 动作空间加上图片就是观测空间
            use_videos=cfg.dataset.video,
        ),
    )

    # 继续收集
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # Load pretrained policy
    print(cfg.policy)
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    # 仿真环境中这里不需要连接
    # robot.connect()
    if teleop is not None:
        teleop.connect()

    # 方向键右键: -> 提前结束当前循环
    # 方向键左键: -> 重新收集此数据
    # Esc: 结束收集
    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        # 仿真和收集数据主循环
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:

            # 收集数据
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
            logging.info("按 -> 提前进入下一条")
            logging.info("按 <- 重新收集当前这条")
            logging.info("按 Esc 结束收集")
            record_loop(
                sim_robot,
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment", cfg.play_sounds)
                logging.info("再次点击 -> ")
                record_loop(
                    sim_robot,
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                sim_robot.reset_sim(manual=True)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                print('rerecord episode and exit early, reset sim')
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    # robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)

    sim_robot.exit_sim()
    return dataset

def main():
    register_third_party_devices()
    record()

if __name__ == "__main__":
    main()
