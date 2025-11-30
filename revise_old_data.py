from tqdm import trange
from util import read_json, save_json
import numpy as np


for episode_index in trange(503):
    state_path = f"episodes/episode_{episode_index}/robot_state.json"
    data = read_json(state_path)
    data["robot_arm_position"] = data.pop("robot_arm_state")
    # catch_state = np.array(state)[:, 3].tolist()
    # catch_frame = catch_state.index(1)
    # assert catch_state[catch_frame - 1] == 0 and catch_state[catch_frame] == 1
    # task_frame = len(state) - 1
    # data['robot_arm_state'] = np.array(state)[:, :3].tolist()
    # data["catch_frame"] = catch_frame
    # data["task_frame"] = task_frame
    save_json(data, state_path)
