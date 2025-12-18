import cv2
import numpy as np
from tqdm import trange
from util import read_json, save_npz
# from util import soft_state
from macro import ROBOT_ARM_SPEED


def prepare_dataset(episode_from, episode_to, dataset_type):
    assert dataset_type in ["train", "validate"]

    img1_list = []
    img2_list = []
    img3_list = []
    current_state_list = [] # only the current positions, might add other input infos later
    gt_list = []

    for episode_index in trange(episode_from, episode_to, ncols=80):
        state_path = f"episodes/episode_{episode_index}/robot_state.json"
        data = read_json(state_path)
        position = data["robot_arm_position"]
        num_frames = len(position)
        for frame_index in range(num_frames):
            img1_path = f"episodes/episode_{episode_index}/camera_1/{frame_index}.png"
            img2_path = f"episodes/episode_{episode_index}/camera_2/{frame_index}.png"
            img3_path = f"episodes/episode_{episode_index}/camera_3/{frame_index}.png"

            img1_list.append(load_and_process_image(img1_path))
            img2_list.append(load_and_process_image(img2_path))
            img3_list.append(load_and_process_image(img3_path))

            if frame_index == num_frames - 1:
                # last frame, no action, already in place
                current_state = [*position[frame_index]]
                ground_truth = [0.0, 0.0, 0.0, 1.0]
            else:
                # not currently in place
                current_position = position[frame_index]
                next_position = position[frame_index + 1]
                current_state = [*current_position]

                delta_x, delta_y, delta_z = (np.array(next_position) - np.array(current_position)).tolist()
                delta_x /= ROBOT_ARM_SPEED # normalize action within range of -1, 1
                delta_y /= ROBOT_ARM_SPEED
                delta_z /= ROBOT_ARM_SPEED
                ground_truth = [delta_x, delta_y, delta_z, 0]
                
            current_state_list.append(current_state)
            gt_list.append(ground_truth)


    data = {
        "img1": np.stack(img1_list),
        "img2": np.stack(img2_list),
        "img3": np.stack(img3_list),
        "current_state": np.array(current_state_list, dtype=np.float16),
        "ground_truth": np.array(gt_list, dtype=np.float16),
    }

    if dataset_type == "train":
        save_npz(data, "data_train.npz")
    else:
        save_npz(data, "data_validate.npz")


def load_and_process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    if img.shape[0] >= 224 and img.shape[1] >= 224:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.transpose(img, [2, 0, 1])
    return img.astype(np.uint8)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="prepare dataset")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["train", "validate"])
    parser.add_argument("--episode_from", type=int, required=True)
    parser.add_argument("--episode_to", type=int, required=True)

    args = parser.parse_args()
    prepare_dataset(episode_from=args.episode_from, episode_to=args.episode_to, dataset_type=args.dataset_type)


# python prepare_dataset.py --episode_from 0 --episode_to 200 --dataset_type validate
# python prepare_dataset.py --episode_from 200 --episode_to 1000 --dataset_type train










# import numpy as np
# from tqdm import trange
# from util import read_json, save_json
# from util import soft_state


# def prepare_dataset(episode_from, episode_to, dataset_type):
#     assert dataset_type in ["train", "validate"]

#     img1_list = []
#     img2_list = []
#     img3_list = []
#     current_state_list = []
#     action_list = []

#     for episode_index in trange(episode_from, episode_to):
#         state_path = f"episodes/episode_{episode_index}/robot_state.json"
#         data = read_json(state_path)
#         position = data["robot_arm_position"]
#         catch_frame = data["catch_frame"]
#         task_frame = data["task_frame"]
#         num_frames = len(position)
#         for frame_index in range(num_frames - 1):
#             img1_path = f"episodes/episode_{episode_index}/camera_1/{frame_index}.png"
#             img2_path = f"episodes/episode_{episode_index}/camera_2/{frame_index}.png"
#             img3_path = f"episodes/episode_{episode_index}/camera_3/{frame_index}.png"
#             img1_list.append(img1_path)
#             img2_list.append(img2_path)
#             img3_list.append(img3_path)

#             current_position = position[frame_index]
#             current_catch = soft_state(frame_index, catch_frame, frame_tolerance=5)
#             current_task = soft_state(frame_index, task_frame, frame_tolerance=5)
#             current_state = [*current_position, current_catch, current_task]

#             next_position = position[frame_index + 1]
#             next_catch = soft_state(frame_index + 1, catch_frame, frame_tolerance=5)
#             next_task = soft_state(frame_index + 2, task_frame, frame_tolerance=5)

#             delta_x, delta_y, delta_z = (np.array(next_position) - np.array(current_position)).tolist()

#             action = [delta_x, delta_y, delta_z, next_catch, next_task]
#             current_state_list.append(current_state)
#             action_list.append(action)

#     action_min = np.min(np.array(action_list)[:, :3], axis=0).tolist()
#     action_max = np.max(np.array(action_list)[:, :3], axis=0).tolist()

#     data = {
#         "img1": img1_list,
#         "img2": img2_list,
#         "img3": img3_list,
#         "current_state": current_state_list,
#         "action": action_list,
#         "action_min": action_min,
#         "action_max": action_max
#     }

#     if dataset_type == "train":
#         save_json(data, "data_train.json")
#     else:
#         save_json(data, "data_validate.json")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="prepare dataset")
#     parser.add_argument("--dataset_type", type=str, required=True, choices=["train", "validate"])
#     parser.add_argument("--episode_from", type=int, required=True)
#     parser.add_argument("--episode_to", type=int, required=True)

#     args = parser.parse_args()
#     prepare_dataset(episode_from=args.episode_from, episode_to=args.episode_to, dataset_type=args.dataset_type)
