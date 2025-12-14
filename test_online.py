import json
import os
import torch
import cv2
import math
import random
import numpy as np
from util import read_json
from scene import SimScene
from model import VisionActionModel
from macro import FLOOR_SIZE, ROBOT_ARM_HEIGHT, ROBOT_ARM_SPEED, DEBUG_MODE


def normalize_image(img, device):
    img = np.transpose(img, [2, 0, 1])
    img = img / 255.0 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32, device=device)
    return img


@torch.inference_mode()
def inference(episode_save_dir, scene, model, device, action_min, action_max):
    # action_min = np.array(action_min, dtype=np.float32)
    # action_max = np.array(action_max, dtype=np.float32)

    # os.makedirs(os.path.join(episode_save_dir, "camera_1"), exist_ok=True)
    # os.makedirs(os.path.join(episode_save_dir, "camera_2"), exist_ok=True)
    # catch_state_record = []
    task_state_record = []

    frame_count = 0
    catch_state = 0  # 1 for catching, 0 for not catching
    task_state = 0  # 1 for complete, 0 for not complete

    while (task_state == 0) and (frame_count < 300):
        # catch_state = 1 if frame_count > 100 else 0
        state = list(scene.robot_arm.location)
        # state.append(catch_state)
        state.append(task_state)
        state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)

        save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
        scene.shot_1(save_path_1)
        img1 = cv2.imread(save_path_1)
        img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
        img1 = normalize_image(img1, device)
        img1 = torch.unsqueeze(img1, dim=0)

        save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
        scene.shot_2(save_path_2)
        img2 = cv2.imread(save_path_2)
        img2 = cv2.resize(img2, (128, 128), interpolation=cv2.INTER_AREA)
        img2 = normalize_image(img2, device)
        img2 = torch.unsqueeze(img2, dim=0)

        save_path_3 = os.path.join(episode_save_dir, "camera_3", f"{frame_count}.png")
        scene.shot_3(save_path_3)
        img3 = cv2.imread(save_path_3)
        img3 = cv2.resize(img3, (128, 128), interpolation=cv2.INTER_AREA)
        img3 = normalize_image(img3, device)
        img3 = torch.unsqueeze(img3, dim=0)

        action = model(img1, img2, img3, state).cpu().numpy().reshape(-1)
        # delta_pos = 0.5 * (action[:3] + 1) * (action_max - action_min) + action_min
        delta_pos = action[:3] * ROBOT_ARM_SPEED
        # next_catch_state = 1 if action[3] > 0.3 else 0
        next_task_state = 1 if action[3] > 0.7 else 0

        # scene.move_robot_arm(dx=delta_pos[0], dy=delta_pos[1], dz=delta_pos[2])
        if scene.update_frame(dx=delta_pos[0], dy=delta_pos[1], dz=delta_pos[2]):
            next_task_state = 1
        
        # if catch_state == 1:
        #     scene.move_object(dx=delta_pos[0], dy=delta_pos[1], dz=delta_pos[2])
        frame_count += 1

        # catch_state = next_catch_state
        task_state = next_task_state

        # catch_state_record.append(action[3].item())
        task_state_record.append(action[3].item())

    data = {
        # "catch_state": catch_state_record,
        "task_state": task_state_record,
        # "object_init_x": scene.object_init_x,
        # "object_init_y": scene.object_init_y,
        "robot_arm_init_x": scene.robot_arm_init_x,
        "robot_arm_init_y": scene.robot_arm_init_y,
        "robot_arm_init_z": scene.robot_arm_init_z
    }

    state_save_path = os.path.join(episode_save_dir, "robot_state.json")
    print(catch_state, task_state, frame_count)
    with open(state_save_path, "w") as f:
        json.dump(data, f)


def main():
    # object_init_x = 8 * random.uniform(-1, 1)
    # object_init_y = 8 * random.uniform(-1, 1)
    robot_arm_init_x = 4 * random.uniform(-1, 1)
    robot_arm_init_y = 4 * random.uniform(-1, 1)
    robot_arm_init_z = random.uniform(6, 10)
    sun_rx_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
    sun_ry_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
    sun_density = 6.0
    bg_r = 1.0
    bg_g = 1.0
    bg_b = 1.0
    bg_density = 0.2

    stat_path = "data_train.json"
    data = read_json(stat_path)
    action_min = data["action_min"]
    action_max = data["action_max"]

    device = "cuda"
    model = VisionActionModel().to(device)
    checkpoint_pth = "checkpoint.pth"
    assert os.path.exists(checkpoint_pth)
    model_state = torch.load(checkpoint_pth)
    model.load_state_dict(model_state)
    model.eval()
    
    will_stick_free = random.uniform(0, 1) > 0.5
    free_angle_percentage = random.triangular(low=0.15, high=0.95, mode=0.9)
    
    # debug
    if DEBUG_MODE:
        will_stick_free = True
        free_angle_percentage = 0.15
        
        
    scene = SimScene(resolution=512, board_init_x=0, board_init_y=0,
                     robot_arm_init_x=0,
                     robot_arm_init_y=0,
                     robot_arm_init_z=3.0,
                     will_stick_free=will_stick_free, free_angle_percentage=free_angle_percentage,
                     sun_rx_radian=sun_rx_radian, sun_ry_radian=sun_ry_radian, sun_density=sun_density,
                     bg_r=bg_r, bg_g=bg_g, bg_b=bg_b, bg_density=bg_density)
    episode_save_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
    inference(episode_save_dir, scene, model, device, action_min, action_max)


if __name__ == "__main__":
    main()






# import json
# import os
# import torch
# import cv2
# import math
# import random
# import numpy as np
# from util import read_json
# from scene import SimScene
# from model import VisionActionModel
# from macro import FLOOR_SIZE, ROBOT_ARM_HEIGHT, ROBOT_ARM_SPEED


# def normalize_image(img, device):
#     img = np.transpose(img, [2, 0, 1])
#     img = img / 255.0 * 2 - 1
#     img = torch.tensor(img, dtype=torch.float32, device=device)
#     return img


# @torch.inference_mode()
# def inference(episode_save_dir, scene, model, device, action_min, action_max):
#     # action_min = np.array(action_min, dtype=np.float32)
#     # action_max = np.array(action_max, dtype=np.float32)

#     os.makedirs(os.path.join(episode_save_dir, "camera_1"), exist_ok=True)
#     os.makedirs(os.path.join(episode_save_dir, "camera_2"), exist_ok=True)
#     catch_state_record = []
#     task_state_record = []

#     frame_count = 0
#     catch_state = 0  # 1 for catching, 0 for not catching
#     task_state = 0  # 1 for complete, 0 for not complete

#     while (task_state == 0) and (frame_count < 300):
#         catch_state = 1 if frame_count > 100 else 0
#         state = list(scene.robot_arm.location)
#         state.append(catch_state)
#         state.append(task_state)
#         state = torch.tensor(state, dtype=torch.float32, device=device).reshape(1, -1)

#         save_path_1 = os.path.join(episode_save_dir, "camera_1", f"{frame_count}.png")
#         scene.shot_1(save_path_1)
#         img1 = cv2.imread(save_path_1)
#         img1 = cv2.resize(img1, (128, 128), interpolation=cv2.INTER_AREA)
#         img1 = normalize_image(img1, device)
#         img1 = torch.unsqueeze(img1, dim=0)

#         save_path_2 = os.path.join(episode_save_dir, "camera_2", f"{frame_count}.png")
#         scene.shot_2(save_path_2)
#         img2 = cv2.imread(save_path_2)
#         img2 = cv2.resize(img2, (128, 128), interpolation=cv2.INTER_AREA)
#         img2 = normalize_image(img2, device)
#         img2 = torch.unsqueeze(img2, dim=0)

#         save_path_3 = os.path.join(episode_save_dir, "camera_3", f"{frame_count}.png")
#         scene.shot_3(save_path_3)
#         img3 = cv2.imread(save_path_3)
#         img3 = cv2.resize(img3, (128, 128), interpolation=cv2.INTER_AREA)
#         img3 = normalize_image(img3, device)
#         img3 = torch.unsqueeze(img3, dim=0)

#         action = model(img1, img2, img3, state).cpu().numpy().reshape(-1)
#         # delta_pos = 0.5 * (action[:3] + 1) * (action_max - action_min) + action_min
#         delta_pos = action[:3] * ROBOT_ARM_SPEED
#         next_catch_state = 1 if action[3] > 0.3 else 0
#         next_task_state = 1 if action[4] > 0.7 else 0

#         scene.move_robot_arm(dx=delta_pos[0], dy=delta_pos[1], dz=delta_pos[2])
#         if catch_state == 1:
#             scene.move_object(dx=delta_pos[0], dy=delta_pos[1], dz=delta_pos[2])
#         frame_count += 1

#         catch_state = next_catch_state
#         task_state = next_task_state

#         catch_state_record.append(action[3].item())
#         task_state_record.append(action[4].item())

#     data = {
#         "catch_state": catch_state_record,
#         "task_state": task_state_record,
#         "object_init_x": scene.object_init_x,
#         "object_init_y": scene.object_init_y,
#         "robot_arm_init_x": scene.robot_arm_init_x,
#         "robot_arm_init_y": scene.robot_arm_init_y,
#         "robot_arm_init_z": scene.robot_arm_init_z
#     }

#     state_save_path = os.path.join(episode_save_dir, "robot_state.json")
#     print(catch_state, task_state, frame_count)
#     with open(state_save_path, "w") as f:
#         json.dump(data, f)


# def main():
#     object_init_x = 8 * random.uniform(-1, 1)
#     object_init_y = 8 * random.uniform(-1, 1)
#     robot_arm_init_x = 4 * random.uniform(-1, 1)
#     robot_arm_init_y = 4 * random.uniform(-1, 1)
#     robot_arm_init_z = random.uniform(6, 10)
#     sun_rx_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
#     sun_ry_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
#     sun_density = 6.0
#     bg_r = 1.0
#     bg_g = 1.0
#     bg_b = 1.0
#     bg_density = 0.2

#     stat_path = "data_train.json"
#     data = read_json(stat_path)
#     action_min = data["action_min"]
#     action_max = data["action_max"]

#     device = "cuda"
#     model = VisionActionModel().to(device)
#     checkpoint_pth = "checkpoint.pth"
#     assert os.path.exists(checkpoint_pth)
#     model_state = torch.load(checkpoint_pth)
#     model.load_state_dict(model_state)
#     model.eval()

#     scene = SimScene(resolution=512, object_init_x=object_init_x, object_init_y=object_init_y,
#                      robot_arm_init_x=robot_arm_init_x,
#                      robot_arm_init_y=robot_arm_init_y,
#                      robot_arm_init_z=robot_arm_init_z,
#                      sun_rx_radian=sun_rx_radian, sun_ry_radian=sun_ry_radian, sun_density=sun_density,
#                      bg_r=bg_r, bg_g=bg_g, bg_b=bg_b, bg_density=bg_density)
#     episode_save_dir = os.path.join(os.path.dirname(__file__), "real_time_test")
#     inference(episode_save_dir, scene, model, device, action_min, action_max)


# if __name__ == "__main__":
#     main()
