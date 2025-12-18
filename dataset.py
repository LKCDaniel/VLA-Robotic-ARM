import numpy as np
import torch
from torch.utils.data import Dataset
from util import load_npz
from macro import ROBOT_ARM_SPEED


# better use npz / pickle for dataset storage and loading

class BendStickDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = load_npz(data_path)

        # Stored images already preprocessed to CHW, float16 in npz; convert to float32 for model input
        self.img1 = torch.from_numpy(data["img1"]) 
        self.img2 = torch.from_numpy(data["img2"])
        self.img3 = torch.from_numpy(data["img3"])
        self.current_state = torch.from_numpy(data["current_state"])
        self.action = torch.from_numpy(data["ground_truth"])
        
        self.length = self.current_state.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        # 3. Cast just-in-time for the model
        # We cast to 'float32' here because standard PyTorch layers are float32.
        # AMP will automatically downcast specific operations to float16/bfloat16 later.
        
        return (
            # Normalize: (0 to 255) -> (-1.0 to 1.0)
            self.img1[index].to(dtype=torch.bfloat16).div_(255.0).mul_(2.0).sub_(1.0),
            self.img2[index].to(dtype=torch.bfloat16).div_(255.0).mul_(2.0).sub_(1.0),
            self.img3[index].to(dtype=torch.bfloat16).div_(255.0).mul_(2.0).sub_(1.0),
            
            # State/Action: float16 (storage) -> float32 (compute input)
            self.current_state[index],
            self.action[index],
        )
        
        return (
            self.img1[index].astype(torch.bfloat16)/255.0 * 2 - 1,
            self.img2[index].astype(torch.bfloat16)/255.0 * 2 - 1,
            self.img3[index].astype(torch.bfloat16)/255.0 * 2 - 1,
            self.current_state[index],
            self.action[index],
        )

