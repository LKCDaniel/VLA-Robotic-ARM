import cv2
import numpy as np
from torch.utils.data import Dataset
from util import read_json
from macro import ROBOT_ARM_SPEED


class PickPlaceDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        data = read_json(json_path)

        self.img1 = data["img1"]
        self.img2 = data["img2"]
        self.img3 = data["img3"]

        self.current_state = np.array(data["current_state"], dtype=np.float32)
        self.action = np.array(data["action"], dtype=np.float32)
        self.action_min = np.array(data["action_min"], dtype=np.float32)
        self.action_max = np.array(data["action_max"], dtype=np.float32)
        self.normalize_action()

        self.length = self.current_state.shape[0]

    def normalize_action(self):
        # action_min = self.action_min.reshape(1, -1)
        # action_max = self.action_max.reshape(1, -1)
        self.action[:, :3] = self.action[:, :3] / ROBOT_ARM_SPEED
        # self.action[:, :3] = (self.action[:, :3] - action_min) / (action_max - action_min) * 2 - 1

    def __len__(self):
        return self.length

    @staticmethod
    def normalize_image(img_path):
        img = cv2.imread(img_path)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.0 * 2 - 1
        return img.astype(np.float32)

    def __getitem__(self, index):
        return (self.normalize_image(self.img1[index]),
                self.normalize_image(self.img2[index]),
                self.normalize_image(self.img3[index]),
                self.current_state[index],
                self.action[index])

