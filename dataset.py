from util import read_json
import cv2
import numpy as np


for episode_index in range(300):
    state_path = f"episodes/episode_{episode_index}/robot_state.json"
    states = read_json(state_path)["robot_arm_state"]
    num_frames = len(states)
    for frame_index in range(num_frames):
        img1_path = f"episodes/episode_{episode_index}/camera_1/{frame_index}.png"
        img2_path = f"episodes/episode_{episode_index}/camera_2/{frame_index}.png"
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        state = np.array(states[frame_index])
        if frame_index < num_frames - 1:
            action = np.zeros(5)
            next_state = np.array(states[frame_index + 1])
            action[:3] = next_state[:3] - state[:3]
            action[3:] = next_state[3:]
        else:
            action = np.array([0, 0, 0, 1, 1])