import cv2
import numpy as np
from torch.utils.data import Dataset
from util import read_json


class PickPlaceDataset(Dataset):
    def __init__(self, json_path, stat_path):
        super().__init__()
        data = read_json(json_path)

        self.img1 = data["img1"]
        self.img2 = data["img2"]
        self.img3 = data["img3"]

        self.current_state = np.array(data["current_state"], dtype=np.float32)
        self.next_state = np.array(data["next_state"], dtype=np.float32)
        self.state_min = np.array(data["state_min"], dtype=np.float32)
        self.state_max = np.array(data["state_max"], dtype=np.float32)
        self.normalize_next_state()

        self.length = self.current_state.shape[0]

    def normalize_next_state(self):
        state_min = self.state_min.reshape(1, -1)
        state_max = self.state_max.reshape(1, -1)
        self.next_state[:, :3] = (self.next_state[:, :3] - state_min) / (state_max - state_min) * 2 - 1

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
                self.next_state[index])

