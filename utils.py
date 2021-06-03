import pickle
import numpy as np
import collections


def load(file_path: str):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def dump(file_path, data):
    file = open(file_path, "wb")
    pickle.dump(data, file)

