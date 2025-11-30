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
    states = read_json(state_path)["robot_arm_state"]
    num_frames = len(states)
    for frame_index in range(num_frames - 1):
        img1_path = f"episodes/episode_{episode_index}/camera_1/{frame_index}.png"
        img2_path = f"episodes/episode_{episode_index}/camera_2/{frame_index}.png"
        img3_path = f"episodes/episode_{episode_index}/camera_3/{frame_index}.png"
        img1_list.append(img1_path)
        img2_list.append(img2_path)
        img3_list.append(img3_path)
        current_state = states[frame_index]
        next_state = states[frame_index + 1]
        delta_x = next_state[0] - current_state[0]
        delta_y = next_state[1] - current_state[1]
        delta_z = next_state[2] - current_state[2]
        action = [delta_x, delta_y, delta_z, next_state[3], next_state[4]]
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

save_json(data, "train_data.json")
