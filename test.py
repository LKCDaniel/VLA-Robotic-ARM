import json
import os
import random
import torch
import cv2
import math
import numpy as np
from util import read_json
from scene import SimScene
from model import VisionActionModel
from macro import FLOOR_SIZE, ROBOT_ARM_HEIGHT, ROBOT_ARM_SPEED


def normalize_image(img, device):
    img = np.transpose(img, [2, 0, 1])
    img = img / 255.0 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32, device=device)
    return img


@torch.inference_mode()
def inference(episode_save_dir, scene, model, device, action_min, action_max):
    action_min = np.array(action_min, dtype=np.float32)
    action_max = np.array(action_max, dtype=np.float32)

    os.makedirs(os.path.join(episode_save_dir, "camera_1"), exist_ok=True)
    os.makedirs(os.path.join(episode_save_dir, "camera_2"), exist_ok=True)
    robot_arm_state_record = []

    frame_count = 0
    catch_state = 0  # 1 for catching, 0 for not catching
    task_state = 0  # 1 for complete, 0 for not complete

    while frame_count < 400:
        state = list(scene.robot_arm.location)
        state.append(catch_state)
        state.append(task_state)
        state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)

        save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
        scene.shot_1(save_path_1)
        img1 = cv2.imread(save_path_1)
        img1 = normalize_image(img1, device)
        img1 = torch.unsqueeze(img1, dim=0)

        save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
        scene.shot_2(save_path_2)
        img2 = cv2.imread(save_path_2)
        img2 = normalize_image(img2, device)
        img2 = torch.unsqueeze(img2, dim=0)

        save_path_3 = os.path.join(episode_save_dir, "camera_3", f"{frame_count}.png")
        scene.shot_3(save_path_3)
        img3 = cv2.imread(save_path_3)
        img3 = normalize_image(img3, device)
        img3 = torch.unsqueeze(img3, dim=0)

        action = model(img1, img2, img3, state).cpu().numpy().reshape(-1)
        # delta_pos = 0.5 * (action[:3] + 1) * (action_max - action_min) + action_min
        delta_pos = action[:3] * ROBOT_ARM_SPEED
        next_catch_state = round(action[3].item())
        next_task_state = round(action[4].item())

        scene.move_robot_arm(dx=delta_pos[0], dy=delta_pos[1], dz=delta_pos[2])
        # if catch_state == 1:
        #     scene.move_object(dx=action[0], dy=action[1], dz=action[2])
        frame_count += 1

        catch_state = next_catch_state
        task_state = next_task_state

    data = {
        "robot_arm_state": robot_arm_state_record,
        "object_init_x": scene.object_init_x,
        "object_init_y": scene.object_init_y
    }

    state_save_path = os.path.join(episode_save_dir, "robot_state.json")
    with open(state_save_path, "w") as f:
        json.dump(data, f)


def main():
    stat_path = "train_data.json"
    data = read_json(stat_path)
    action_min = data["action_min"]
    action_max = data["action_max"]
    device = "cuda"
    model = VisionActionModel().to(device)
    model.eval()
    object_init_x = -2
    object_init_y = -2
    robot_arm_init_x = -4
    robot_arm_init_y = -4
    robot_arm_init_z = 6
    sun_rx_radian = math.pi / 3
    sun_ry_radian = 2 * math.pi / 3
    sun_density = 6.0
    bg_r = 1.0
    bg_g = 1.0
    bg_b = 1.0
    bg_density = 0.2
    scene = SimScene(object_init_x=object_init_x, object_init_y=object_init_y,
                     robot_arm_init_x=robot_arm_init_x,
                     robot_arm_init_y=robot_arm_init_y,
                     robot_arm_init_z=robot_arm_init_z,
                     sun_rx_radian=sun_rx_radian, sun_ry_radian=sun_ry_radian, sun_density=sun_density,
                     bg_r=bg_r, bg_g=bg_g, bg_b=bg_b, bg_density=bg_density)
    episode_save_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
    inference(episode_save_dir, scene, model, device, action_min, action_max)


if __name__ == "__main__":
    main()
