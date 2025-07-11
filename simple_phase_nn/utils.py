import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import itertools
import inspect
import random


sns.set_theme()
sns.color_palette("Paired")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def create_path(folder_path, overwrite=False):
    """
    Creates a path to save a file in a folder.
    If file_name exists, adds a suffix.
    """
    base_folder = Path(folder_path)
    folder = base_folder

    if not overwrite:      
        counter = 1
        # Incrementally check for existing folders and add suffixes
        while folder.exists():
            print(f"Folder {folder} exists. Adding suffix.")
            folder = base_folder.parent / f"{base_folder.name}_{counter}"
            counter += 1

    folder.mkdir(parents=True, exist_ok=True)

    return folder

def func_name():
    return inspect.stack()[1].function


def save_json(obj, folder_path, file_name):
    """
    Saves obj to path as a json file.
    """
    folder = Path(folder_path)
    load_path = folder.joinpath(file_name)
    with open(load_path, "w") as file:
        json.dump(obj, file)

def load_json(folder_path, file_name):
    """
    Loads a json file from path.
    """
    folder = Path(folder_path)
    load_path = folder.joinpath(file_name)
    with open(load_path) as file:
        return json.load(file)


def set_seeds(seed_no):
    """"
    Sets the seed for reproducibility.
    NB: seed is accessible globally.
    """
    global seed
    seed = seed_no
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EarlyStopper:
    """
    Standard early stopping class.
    """
    def __init__(self, patience=1, min_delta=0, verbose=True, relative=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_obj = float('inf')
        self.verbose = verbose
        self.relative = relative

    def early_stop(self, validation_obj):
        if np.isnan(validation_obj):
            print("Validation objective is NaN. Stopping early.")
            return True
        difference = validation_obj - self.min_validation_obj
        if self.relative:
            difference /= self.min_validation_obj
        if validation_obj < self.min_validation_obj:
            if self.verbose:
                print(f"Validation objective decreased ({self.min_validation_obj:.2e} --> {validation_obj:.2e}).")
            self.min_validation_obj = validation_obj
            self.counter = 0
        elif difference >= self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"Validation objective increased ({self.min_validation_obj:.2e} --> {validation_obj:.2e}).")
                print(f"Early stopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                return True
        return False




def normalize01(y, y_min, y_max):
    """
    Normalize to [0, 1] range.
    """
    return (y - y_min) / (y_max - y_min)

def denormalize01(y, y_min, y_max):
    """
    Denormalize from [0, 1] range to original range.
    """
    return y * (y_max - y_min) + y_min




