import numpy as np
import json


def read_json(p):
    with open(p, "r") as f:
        return json.load(f)


def compute_normal_component(base_vector, vector):
    unit_base_vector = base_vector / np.linalg.norm(base_vector)
    dot_product = np.dot(vector, unit_base_vector)
    return vector - dot_product * unit_base_vector


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def compute_distance(pos1, pos2):
    distance = np.linalg.norm(pos2 - pos1)
    return distance
