import json
import os
import random
from scene import SimScene
import numpy as np
from util import normalize_vector


def record_episode(episode_save_dir, scene):
    os.makedirs(os.path.join(episode_save_dir, "camera_1"), exist_ok=True)
    os.makedirs(os.path.join(episode_save_dir, "camera_2"), exist_ok=True)
    frame_count = 0
    catch_state = 0  # 1 for catching, 0 for not catching
    robot_arm_speed = 0.1
    robot_arm_state_record = []
    while True:
        print(scene.robot_arm.location)
        save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
        scene.shot_1(save_path_1)
        save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
        scene.shot_2(save_path_2)
        robot_arm_state = list(scene.robot_arm.location)
        robot_arm_state.append(catch_state)
        robot_arm_state_record.append(robot_arm_state)
        robot_arm_to_pick_object = scene.robot_arm_to_pick_object()
        if np.linalg.norm(robot_arm_to_pick_object) < robot_arm_speed * 1.1:
            break
        action = robot_arm_speed * normalize_vector(robot_arm_to_pick_object)
        scene.move_robot_arm(dx=action[0], dy=action[1], dz=action[2])
        frame_count += 1
    catch_state = 1
    data = {
        "robot_arm_state": robot_arm_state_record,
        "object_init_x": scene.object_init_x,
        "object_init_y": scene.object_init_y
    }
    state_save_path = os.path.join(episode_save_dir, "robot_state.json")
    with open(state_save_path, "w") as f:
        json.dump(data, f)


def main():
    init_object_x = random.uniform(-3, 3)
    init_object_y = random.uniform(-3, 3)
    scene = SimScene(object_init_x=init_object_x, object_init_y=init_object_y)
    episode_index = 0
    episode_save_dir = os.path.join(os.path.dirname(__file__), "episodes", f"episode_{episode_index}")
    record_episode(episode_save_dir, scene)


if __name__ == "__main__":
    main()
