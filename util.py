import numpy as np
import json
import math
import torch
import psutil
import os
import sys

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure data is written immediately

    def flush(self):
        # Needed for python 3 compatibility
        self.terminal.flush()
        self.log.flush()

def check_memory_status():
        print(f"\nGPU memory: {round(torch.cuda.memory_allocated() / (1024 ** 3), 2)} GB allocated, {round(torch.cuda.memory_reserved() / (1024 **3), 2)} GB reserved.")
        
        # print(torch.cuda.memory_summary(device='cuda', abbreviated=True)) # print a chart

        # --- SYSTEM RAM (CPU) ---
        # 1. Get overall system memory stats
        sys_mem = psutil.virtual_memory()
        total_ram = sys_mem.total / (1024 ** 3)
        used_ram = sys_mem.used / (1024 ** 3)
        percent_ram = sys_mem.percent
        
        # 2. Get memory used specifically by this Python script
        process = psutil.Process(os.getpid())
        process_mem = process.memory_info().rss / (1024 ** 3)  # RSS = Resident Set Size (physical memory)

        print(f"System RAM : {used_ram:.2f} GB used / {total_ram:.2f} GB total ({percent_ram}%)")
        print(f"Script RAM : {process_mem:.2f} GB used by this process\n")

def load_npz(p):
    return np.load(p, allow_pickle=False)


def save_npz(data, p):
    np.savez_compressed(p, **data)


def read_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(data, p):
    with open(p, "w") as f:
        json.dump(data, f)


def compute_normal_component(base_vector, vector):
    unit_base_vector = base_vector / np.linalg.norm(base_vector)
    dot_product = np.dot(vector, unit_base_vector)
    return vector - dot_product * unit_base_vector


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def compute_distance(pos1, pos2):
    distance = np.linalg.norm(pos2 - pos1)
    return distance


# # it is not a good practice to rely on such transition approach for logit prediction (task completion)
# def soft_state(current_frame, target_frame, frame_tolerance):
#     result = 1 + (current_frame - target_frame) / frame_tolerance
#     result = min(result, 1)
#     result = max(result, 0)
#     return result
