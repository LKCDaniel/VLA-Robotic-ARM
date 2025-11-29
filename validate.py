import json
import os
import random
import torch
import cv2
import numpy as np
from util import read_json
from scene import SimScene
from model import VisionActionModel


def normalize_image(img, device):
    img = np.transpose(img, [2, 0, 1])
    img = img / 255.0 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32, device=device)
    return img


@torch.inference_mode()
def inference(episode_save_dir, scene, model, device, state_min, state_max):
    state_min = np.array(state_min, dtype=np.float32)
    state_max = np.array(state_max, dtype=np.float32)

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

        next_state = model(img1, img2, state).cpu().numpy().reshape(-1)
        next_pos = 0.5 * (next_state[:3] + 1) * (state_max - state_min) + state_min
        next_catch_state = round(next_state[3].item())
        next_task_state = round(next_state[4].item())

        scene.set_robot_arm_location(x=next_pos[0], y=next_pos[1], z=next_pos[2])
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
    state_min = data["state_min"]
    state_max = data["state_max"]
    device = "cuda"
    model = VisionActionModel().to(device)
    model.eval()
    init_object_x = random.uniform(-4, 4)
    init_object_y = random.uniform(-4, 4)
    scene = SimScene(object_init_x=init_object_x, object_init_y=init_object_y)
    episode_save_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
    inference(episode_save_dir, scene, model, device, state_min, state_max)


if __name__ == "__main__":
    main()
