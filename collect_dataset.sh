python -m record.lerobot_record_sim_dataset \
    --model_path=model/trs_so_arm100/scene.xml \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, side: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30,fourcc: "MJPG"}}" \
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