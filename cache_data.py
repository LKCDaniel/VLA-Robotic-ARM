import numpy as np
from tqdm import trange
from util import read_json, save_json

img1_list = []
img2_list = []
img3_list = []
current_state_list = []
action_list = []

for episode_index in trange(400):
    state_path = f"episodes/episode_{episode_index}/robot_state.json"
    data = read_json(state_path)
    position = data["robot_arm_position"]
    catch_frame = data["catch_frame"]
    task_frame = data["task_frame"]
    num_frames = len(position)
    for frame_index in range(num_frames - 1):
        img1_path = f"episodes/episode_{episode_index}/camera_1/{frame_index}.png"
        img2_path = f"episodes/episode_{episode_index}/camera_2/{frame_index}.png"
        img3_path = f"episodes/episode_{episode_index}/camera_3/{frame_index}.png"
        img1_list.append(img1_path)
        img2_list.append(img2_path)
        img3_list.append(img3_path)

        current_position = position[frame_index]
        current_catch = 1 if (frame_index >= catch_frame - 2) else 0
        current_task = 1 if (frame_index >= task_frame - 2) else 0
        current_state = [*current_position, current_catch, current_task]

        next_position = position[frame_index + 1]
        next_catch = 1 if (frame_index + 1 >= catch_frame - 2) else 0
        next_task = 1 if (frame_index + 1 >= task_frame - 2) else 0

        delta_x, delta_y, delta_z = (np.array(next_position) - np.array(current_position)).tolist()

        action = [delta_x, delta_y, delta_z, next_catch, next_task]
        current_state_list.append(current_state)
        action_list.append(action)

action_min = np.min(np.array(action_list)[:, :3], axis=0).tolist()
action_max = np.max(np.array(action_list)[:, :3], axis=0).tolist()

data = {
    "img1": img1_list,
    "img2": img2_list,
    "img3": img3_list,
    "current_state": current_state_list,
    "action": action_list,
    "action_min": action_min,
    "action_max": action_max
}

save_json(data, "data_train.json")
