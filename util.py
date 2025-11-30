import numpy as np
import json
import math


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


def soft_state(current_frame, target_frame, frame_tolerance):
    result = 1 + (current_frame - target_frame) / frame_tolerance
    result = min(result, 1)
    result = max(result, 0)
    return result
