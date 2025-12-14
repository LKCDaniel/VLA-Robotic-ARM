import json
import os
import random
from scene import SimScene
import numpy as np
from util import normalize_vector, save_json
from macro import FLOOR_SIZE, ROBOT_ARM_HEIGHT, ROBOT_ARM_SPEED, DEBUG_MODE
import math


def record_episode(episode_save_dir, scene):
    os.makedirs(os.path.join(episode_save_dir, "camera_1"), exist_ok=True)
    os.makedirs(os.path.join(episode_save_dir, "camera_2"), exist_ok=True)
    os.makedirs(os.path.join(episode_save_dir, "camera_3"), exist_ok=True)
    robot_arm_position_list = []

    frame_count = 0
    free_frame = -1
    # touching_list = []
    temp_variable = scene.will_stick_free

    while True:
        if not DEBUG_MODE: # debug
            save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
            scene.shot_1(save_path_1)
            save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
            scene.shot_2(save_path_2)
        save_path_3 = os.path.join(episode_save_dir, "camera_3", f"{frame_count}.png")
        scene.shot_3(save_path_3)

        robot_arm_position_list.append(list(scene.robot_arm.location))
        # touching_list.append(scene.touch_score())
        if temp_variable and not scene.will_stick_free:
            free_frame = frame_count
            temp_variable = False

        action = scene.next_robot_arm_movement() * ROBOT_ARM_SPEED
        frame_count += 1
        if scene.update_frame(dx=action[0], dy=action[1], dz=action[2]):
            break

    data = {
        "robot_arm_position": robot_arm_position_list,
        # "board_init_x": scene.board_init_x,
        # "board_init_y": scene.board_init_y,
        "robot_arm_init_x": scene.robot_arm_init_x,
        "robot_arm_init_y": scene.robot_arm_init_y,
        "robot_arm_init_z": scene.robot_arm_init_z,
        "free_angle_percentage": (scene.free_angle_degrees-120) / 60,
        "free_frame": free_frame, # -1 if never free
        # "touching_list": touching_list,
        "task_frame": frame_count
    }

    state_save_path = os.path.join(episode_save_dir, "robot_state.json")
    save_json(data, state_save_path)


def generate_episodes(episode_from, episode_to, resolution):
    for episode_index in range(episode_from, episode_to):
        # board_init_x = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
        # board_init_y = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
        # robot_arm_init_x = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
        # robot_arm_init_y = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
        # robot_arm_init_z = random.uniform(ROBOT_ARM_HEIGHT * 0.5, 15)


        sun_rx_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
        sun_ry_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
        # bg_r = random.uniform(0.9, 1.0)
        # bg_g = random.uniform(0.9, 1.0)
        # bg_b = random.uniform(0.7, 1.0)
        # bg_dense = random.uniform(0.2, 0.3)
        sun_density = 6.0
        bg_r = 1.0
        bg_g = 1.0
        bg_b = 1.0
        bg_density = 0.2
        
        will_stick_free = random.uniform(0, 1) > 0.5
        free_angle_percentage = random.triangular(low=0.4, high=0.95, mode=0.9)
        
        if DEBUG_MODE:
            will_stick_free = True # debug
            free_angle_percentage = 0.4 # debug
        
        scene = SimScene(resolution=resolution,
                         board_init_x=0, board_init_y=0,
                         robot_arm_init_x=0,
                         robot_arm_init_y=0,
                         robot_arm_init_z=3.0,
                         will_stick_free=will_stick_free, free_angle_percentage=free_angle_percentage,
                         sun_rx_radian=sun_rx_radian, sun_ry_radian=sun_ry_radian, sun_density=sun_density,
                         bg_r=bg_r, bg_g=bg_g, bg_b=bg_b, bg_density=bg_density)
        
        episode_save_dir = os.path.join(os.path.dirname(__file__), "episodes", f"episode_{episode_index}")
        record_episode(episode_save_dir, scene)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="prepare dataset")
    parser.add_argument("--episode_from", type=int, required=True, default=0)
    parser.add_argument("--episode_to", type=int, required=True, default=500)
    parser.add_argument("--resolution", type=int, required=True, default=256)

    args = parser.parse_args()
    generate_episodes(episode_from=args.episode_from, episode_to=args.episode_to, resolution=args.resolution)

# python record_episode.py --episode_from 0 --episode_to 1 --resolution 256






# import json
# import os
# import random
# from scene import SimScene
# import numpy as np
# from util import normalize_vector, save_json
# from macro import FLOOR_SIZE, ROBOT_ARM_HEIGHT, ROBOT_ARM_SPEED
# import math


# def record_episode(episode_save_dir, scene):
#     os.makedirs(os.path.join(episode_save_dir, "camera_1"), exist_ok=True)
#     os.makedirs(os.path.join(episode_save_dir, "camera_2"), exist_ok=True)
#     os.makedirs(os.path.join(episode_save_dir, "camera_3"), exist_ok=True)
#     robot_arm_position_list = []

#     frame_count = 0

#     while True:
#         robot_arm_to_pick_object = scene.robot_arm_to_pick_object()
#         if np.linalg.norm(robot_arm_to_pick_object) < ROBOT_ARM_SPEED:
#             break

#         save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
#         scene.shot_1(save_path_1)
#         save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
#         scene.shot_2(save_path_2)
#         save_path_3 = os.path.join(episode_save_dir, "camera_3", f"{frame_count}.png")
#         scene.shot_3(save_path_3)

#         robot_arm_position_list.append(list(scene.robot_arm.location))

#         action = ROBOT_ARM_SPEED * normalize_vector(robot_arm_to_pick_object)
#         scene.move_robot_arm(dx=action[0], dy=action[1], dz=action[2])
#         frame_count += 1

#     catch_frame = frame_count

#     while True:
#         robot_arm_to_place_object = scene.robot_arm_to_place_object()
#         if np.linalg.norm(robot_arm_to_place_object) < ROBOT_ARM_SPEED:
#             break

#         save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
#         scene.shot_1(save_path_1)
#         save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
#         scene.shot_2(save_path_2)
#         save_path_3 = os.path.join(episode_save_dir, "camera_3", f"{frame_count}.png")
#         scene.shot_3(save_path_3)

#         robot_arm_position_list.append(list(scene.robot_arm.location))

#         action = ROBOT_ARM_SPEED * normalize_vector(robot_arm_to_place_object)
#         scene.move_robot_arm(dx=action[0], dy=action[1], dz=action[2])
#         scene.move_object(dx=action[0], dy=action[1], dz=action[2])
#         frame_count += 1

#     task_frame = frame_count - 1
#     robot_arm_position_list[-1][-1] = 1
#     data = {
#         "robot_arm_position": robot_arm_position_list,
#         "object_init_x": scene.object_init_x,
#         "object_init_y": scene.object_init_y,
#         "robot_arm_init_x": scene.robot_arm_init_x,
#         "robot_arm_init_y": scene.robot_arm_init_y,
#         "robot_arm_init_z": scene.robot_arm_init_z,
#         "catch_frame": catch_frame,
#         "task_frame": task_frame
#     }

#     state_save_path = os.path.join(episode_save_dir, "robot_state.json")
#     save_json(data, state_save_path)


# def generate_episodes(num_episodes, resolution):
#     for episode_index in range(num_episodes):
#         object_init_x = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
#         object_init_y = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
#         robot_arm_init_x = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
#         robot_arm_init_y = FLOOR_SIZE * 0.5 * random.uniform(-1, 1)
#         robot_arm_init_z = random.uniform(ROBOT_ARM_HEIGHT * 0.5, 15)
#         sun_rx_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
#         sun_ry_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
#         # bg_r = random.uniform(0.9, 1.0)
#         # bg_g = random.uniform(0.9, 1.0)
#         # bg_b = random.uniform(0.7, 1.0)
#         # bg_dense = random.uniform(0.2, 0.3)
#         sun_density = 6.0
#         bg_r = 1.0
#         bg_g = 1.0
#         bg_b = 1.0
#         bg_density = 0.2
#         scene = SimScene(resolution=resolution, object_init_x=object_init_x, object_init_y=object_init_y,
#                          robot_arm_init_x=robot_arm_init_x,
#                          robot_arm_init_y=robot_arm_init_y,
#                          robot_arm_init_z=robot_arm_init_z,
#                          sun_rx_radian=sun_rx_radian, sun_ry_radian=sun_ry_radian, sun_density=sun_density,
#                          bg_r=bg_r, bg_g=bg_g, bg_b=bg_b, bg_density=bg_density)
#         episode_save_dir = os.path.join(os.path.dirname(__file__), "episodes", f"episode_{episode_index}")
#         record_episode(episode_save_dir, scene)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="prepare dataset")
#     parser.add_argument("--num_episodes", type=int, required=True, default=500)
#     parser.add_argument("--resolution", type=int, required=True, default=128)

#     args = parser.parse_args()
#     generate_episodes(num_episodes=args.num_episodes, resolution=args.resolution)