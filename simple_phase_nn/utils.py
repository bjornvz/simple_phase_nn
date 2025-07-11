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
    
##############################################
# TetrisCNN specific functions
##############################################


def get_all_4point_correlators(A):
    """
    Compute all 70 4-point correlators from the input tensor A.
    Args: A (torch.Tensor): Input tensor of shape (N, C, H, W), where C=2.
    Returns: torch.Tensor: A tensor of shape (N, 70) containing the correlators for each batch.
    """
    assert A.dim() == 4, "Input must be a 4D tensor (N, C, H, W)"
    assert A.shape[2] > 1 and A.shape[3] > 1, "Each matrix must be at least 2x2 in spatial dimensions"
    # Generate the 8 shifted tensors
    shifts = [
        A[:, 0, :, :],
        torch.nn.functional.pad(A[:, 0, :, 1:], (0, 1, 0, 0)),
        torch.nn.functional.pad(A[:, 0, 1:, :], (0, 0, 0, 1)),
        torch.nn.functional.pad(A[:, 0, 1:, 1:], (0, 1, 0, 1)),
        A[:, 1, :, :],
        torch.nn.functional.pad(A[:, 1, :, 1:], (0, 1, 0, 0)),
        torch.nn.functional.pad(A[:, 1, 1:, :], (0, 0, 0, 1)),
        torch.nn.functional.pad(A[:, 1, 1:, 1:], (0, 1, 0, 1)),
    ]

    # Compute all 70 4-point correlators
    correlators = torch.stack([
        (a * b * c * d).mean(dim=(1, 2))
        for a, b, c, d in itertools.combinations(shifts, 4)
    ], dim=1)  # Shape: (B, 70)

    return correlators


def get_correlator_origin(index):
    """
    Given an index, find the corresponding correlator shifts.
    """
    # Define shift names in order
    shift_names = [
        "A000", "A001", "A010", "A011",
        "A100", "A101", "A110", "A111",
    ]

    # All 70 combinations of 4 shifts from 8
    combinations = list(itertools.combinations(shift_names, 4))

    correlator_shifts = combinations[index]
    print(f"Correlator {index} uses shifts: {correlator_shifts}")


def l1_regularization(z, penalties):
    """"
    Returns $\sum_{k=1}^K |z_k| \lambda_k$, where $z_k$ is the $k-$th bottleneck activation,
    and $\lambda_k$ is the corresponding penalty for this branch. l1 regularization encourages
    sparsity in the bottleneck, and penalties encourage simplicity.
    """
    assert z.shape[1] == len(penalties), "Number of penalties must match the number of branches."
    branches = torch.abs(z)
    branches *= penalties
    return torch.sum( branches , dim=1 ) # sum over the branches



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




def get_branch_penalties(lambda_params, kernels):
    """"
    Calculates the penalties (lambdas) for network branches, based on kernel parameters.
    Large kernels (areas) are penalized more; i.e. have larger lambda values, exponentially.
    Kernels with more dilation are penalized more, linearly.
    
    lambdas[0]: base of exponential, default 10
    lambdas[1]: min exponent, default -4
    lambdas[2]: max exponent, default 0
    lambdas[3]: n_pen, default 1. 
    This is penalty scaling for additional kernels of the same area. 
    Models trained with this are marked as `VarLambdas`.
    For model without this, set n_pen to 1.
    
    """
    max_kernel_area = max([k[0][0] * k[0][1] for k in kernels]) # e.g. 3x1=3 for TFIM

    penalty_base = np.logspace (
        base = float(lambda_params[0]),
        start = float(lambda_params[1]),
        stop = float(lambda_params[2]),
        num = int(max_kernel_area),
    )
    n_pen = float(lambda_params[3])

    penalties = []
    for k_shape, k_filt_num, k_dilation, k_mask in kernels: # TODO! need to account for mask
        if k_mask:
            k_area = np.count_nonzero(np.array(k_mask))
        else:
            k_area = k_shape[0] * k_shape[1]

        l = penalty_base[k_area - 1] # e.g. area 1 => 10^{lambda_params[1]} = 10^{\lambda_{min}}
        # TODO: what if areas are not consecutive, e.g. from 2x1=2 to 2x2=4? 
        
        for fnum in range(k_filt_num):
            factor = k_dilation if k_area > 1 else 1 # linear scaling with dilation
            scale = n_pen**fnum if n_pen != 1 else 1 # exponential scaling with number of filters, only if n_pen != 1
            penalties.append(l * scale * factor)
    return np.array(penalties).flatten()

def get_ILGT_kernels(filt_num=1):
    """"
    Predefined list of kernels for ILGT as defined by Kacper.
    """
    kernels = [
        # Format is [(kernel height, kernel width), filter number, dilation]
        [(1, 1), filt_num, 1],
        [(2, 1), filt_num, 1],
        [(1, 2), filt_num, 1],
        [(2, 2), filt_num, 1],
        [(2, 1), filt_num, 2],
        [(1, 2), filt_num, 2],
        [(3, 1), filt_num, 1],
        [(3, 2), filt_num, 1],
        [(1, 3), filt_num, 1],
        [(2, 3), filt_num, 1],
        [(3, 3), filt_num, 1],
    ]
    return kernels
