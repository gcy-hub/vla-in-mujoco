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