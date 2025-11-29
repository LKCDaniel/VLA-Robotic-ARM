import numpy as np
from torch.utils.data import Dataset


class PickPlaceDataset(Dataset):
    def __init__(self, npz_path, stat_path):
        super().__init__()
        data = np.load(npz_path)

        self.img1 = data["img1"]
        self.img2 = data["img2"]

        self.state = data["state"].astype(np.float32)
        self.state_min = np.min(self.state, axis=0)
        self.state_max = np.max(self.state, axis=0)
        self.normalize_state()

        self.action = data["action"].astype(np.float32)
        self.action_min = np.min(self.action, axis=0)
        self.action_max = np.max(self.action, axis=0)
        self.normalize_action()

        self.length = self.state.shape[0]
        self.save_state(stat_path)

    def normalize_state(self):
        state_min = self.state_min.reshape(1, -1)
        state_max = self.state_max.reshape(1, -1)
        self.state = (self.state - state_min) / (state_max - state_min) * 2 - 1

    def normalize_action(self):
        action_min = self.action_min.reshape(1, -1)
        action_max = self.action_max.reshape(1, -1)
        self.action = (self.action - action_min) / (action_max - action_min) * 2 - 1

    def __len__(self):
        return self.length

    @staticmethod
    def normalize_image(img):
        img = img / 255.0 * 2 - 1
        return img.astype(np.float32)

    def save_state(self, stat_path):
        print("state_min", self.state_min)
        print("state_max", self.state_max)
        print("action_min", self.action_min)
        print("action_max", self.action_max)
        np.savez(stat_path,
                 state_min=self.state_min,
                 state_max=self.state_max,
                 action_min=self.action_min,
                 action_max=self.action_max)

    def __getitem__(self, index):
        return self.normalize_image(self.img1[index]), self.normalize_image(self.img2[index]), self.state[index], self.action[index]

