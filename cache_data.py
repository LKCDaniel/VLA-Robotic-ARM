from util import read_json
import cv2
import numpy as np
from tqdm import trange


img1_list = []
img2_list = []
state_list = []
action_list = []


for episode_index in trange(300):
    state_path = f"episodes/episode_{episode_index}/robot_state.json"
    states = read_json(state_path)["robot_arm_state"]
    num_frames = len(states)
    for frame_index in range(num_frames):
        img1_path = f"episodes/episode_{episode_index}/camera_1/{frame_index}.png"
        img2_path = f"episodes/episode_{episode_index}/camera_2/{frame_index}.png"
        img1 = cv2.imread(img1_path)
        img1 = np.transpose(img1, [2, 0, 1])
        img1_list.append(img1)
        img2 = cv2.imread(img2_path)
        img2 = np.transpose(img2, [2, 0, 1])
        img2_list.append(img2)
        state = np.array(states[frame_index])
        state_list.append(state)
        if frame_index < num_frames - 1:
            action = np.zeros(5)
            next_state = np.array(states[frame_index + 1])
            action[:3] = next_state[:3] - state[:3]
            action[3:] = next_state[3:]
        else:
            action = np.array([0, 0, 0, 1, 1])
        action_list.append(action)

img1_list = np.array(img1_list)
img2_list = np.array(img2_list)
state_list = np.array(state_list)
action_list = np.array(action_list)
np.savez("data.npz",
         img1=img1_list,
         img2=img2_list,
         state=state_list,
         action=action_list)
