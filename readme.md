# 🚀 VLA in Mujoco

> mujoco仿真平台收集lerobot数据训练VLA。

## ✨ 目录  
- [关于项目](#关于项目)  
- [快速开始](#快速开始)  
  - [环境](#环境)  
  - [安装](#安装)  
  - [运行示例](#运行示例)  

## 关于项目
- 本项目的初衷是让像作者一样入门VLA的小白，也能够从数据收集到模型训练和验证，体验VLA模型;
- mujoco仿真平台安装方便，对小白十分友好;  
- 目前测试了SO-ARM100机械臂、pi05模型，测试demo是使用机械臂推开门;  
- 本项目最大的问题就是模型之间物理碰撞还没有理清 (作者刚入门mujoco T_T).  
**本来还有使用XBox手柄控制机械臂的，但是效果不佳，就舍弃了**  

目前实现的功能有：在mujoco环境中收集VLA训练数据，使用lerobot框架训练、验证=

## 快速开始  
### 环境  
本人环境：  
- Ubuntu 24.04
- Python 3.10   
- CUDA 12.8
- Nvidia RTX 3090 (pi05推理)
- L40 (pi05训练, 远程服务器)   

硬件：  
so-arm101 主臂 (已校准，教程见： [Seeed机械臂校准教程](https://wiki.seeedstudio.com/cn/lerobot_so100m_new/))

### 安装
```bash
# 创建conda环境
conda create -y -n vla_in_mujoco python=3.10 && conda activate vla_in_mujoco

# 视频处理
conda install -y ffmpeg=7.1.1 -c conda-forge

# torch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 其他依赖库
pip install -r requirements.txt

# 安装舵机驱动库
pip install 'lerobot[feetch]'

# 如果要使用pi05训练
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"

# 如果要用smolvla训练
pip install 'lerobot[smolvla]'

```

### 运行示例
so-arm101主臂控制仿真机械臂运动，运行`so_arm_101_controll_demo.py`   
**运行前提：** 机械臂已校准，然后上电，并使能Ubuntu端口
```
from env import  MultiViewerWithLeader

# 场景文件
SCENE_XML_PATH = 'model/trs_so_arm100/scene.xml'

def multi_view_with_so_arm101_demo(model_path, teleop_id, teleop_port):
    """
    model_path： 上面的场景文件
    teleop_id： 机械臂校准时的id, 也就是在so_arm101校准教程中使用的teleop.id
    teleop_port: 机械臂驱动器插入电脑的USB端口号
    """
    viewer = MultiViewerWithLeader(model_path, teleop_id=teleop_id, teleop_port=teleop_port)
    viewer.run_loop()
# 这里的参数仅供参考，视具体情况而定
multi_view_with_so_arm101_demo(SCENE_XML_PATH, 'my_awesome_leader_arm', '/dev/ttyACM1')
```

so-arm101主臂控制仿真机械臂运动并收集数据，运行`./collect_dataset.sh`, 下面是脚本内容，可能需要修改的参数是：  
+ model_path： 场景文件
+ teleop.port： 根据自己的插入的端口号来确定  
+ dataset.num_episodes： 收集轨迹总数量
+ dataset.single_task： VLA任务文本
+ dataset.episode_time_s： 单条轨迹收集时间
+ dataset.reset_time_s： 重启环境时间
+ resume： 继续收集之前没有收集完的数据集
```
python -m record.lerobot_record_sim_dataset \
    --model_path=model/trs_so_arm100/scene.xml \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=my_data/sim_test \
    --dataset.num_episodes=50 \
    --dataset.single_task="Open the slide cabinet’s door" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 \
#    --resume=true
```

训练  
训练和教程里面一样，本项目主要重视数据集收集，训练教程参考[官方](https://github.com/huggingface/lerobot)或者[seeed教程](https://wiki.seeedstudio.com/cn/lerobot_so100m_new/)

验证  
运行`./eval_pi05.sh`, `robot.cameras`保留和收集数据时一样的参数即可
```
python -m record.lerobot_record_sim_dataset \
  --model_path=model/trs_so_arm100/scene.xml \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30,fourcc: "MJPG"}}" \
  --robot.id=my_awesome_follower_arm \
  --display_data=true \
  --dataset.repo_id=my_data/eval_pi05_sim_test \
  --dataset.single_task="Open the slide cabinet’s door" \
  --policy.path=outputs/pi05_training_sim_test/checkpoints/080000/pretrained_model \
  --dataset.episode_time_s=9999 \
  --dataset.reset_time_s=0 \
  --dataset.push_to_hub=false
```


