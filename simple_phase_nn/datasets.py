import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import os
import re

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from simple_phase_nn.utils import *


class PhaseDataset(Dataset, ABC):
    """"
    Base class for phase discovery datasets. 
    - discrete_labels property; used to toggle regression/classification in the training loop.
    - get_phase_indicator method; computes a phase indicator, method depends on regression/classification task.
    """
    @property
    @abstractmethod
    def discrete_labels(self):
        pass

    @abstractmethod
    def get_phase_indicator(self):
        """
        Computes a phase indicator based on model output and configuration.
        This method should be implemented in subclasses.
        """
        pass


 

class XXZDataset(PhaseDataset):

    def __init__(self, train, normalize=True):
        self._discrete_labels = False
        self.no_classes = 1
        self.normalize = normalize

        grid_size = (300,) 
        directory = './data/simulated/XXZ/'
        file_list = [os.path.join(directory, f) for f in os.listdir(directory)]

        # Loads files that start with 't=' and end with '.dat' in the specified directory
        self.samples = []

        for filepath in file_list:
            filename = os.path.basename(filepath)
            print(f"Processing file: {filename}")
            if filename.startswith('Sz') and filename.endswith('.txt'):
                match = re.search(r'Jz([-+]?[0-9]*\.?[0-9]+)', filename)
                if match:
                    label = float(match.group(1))
                    data = np.loadtxt(filepath)
                    reshaped = data.reshape(grid_size[0],)
                    print(f"label: {label}, data: {reshaped.shape}")
                    self.samples.append((
                                        torch.tensor(reshaped, dtype=torch.float32).unsqueeze(0),
                                        torch.tensor(label, dtype=torch.float32)
                                    ))   

        np.random.shuffle(self.samples)  # NOTE: averaging over random seeds will be done using different splits; careful with hyperparameter tuning!
        split = int(0.7 * len(self.samples))
        self.samples = self.samples[:split] if train else self.samples[split:]

        inputs, labels = zip(*self.samples)
        labels = torch.tensor(labels, dtype=torch.float32)
        self.unique_labels_unnormalized = np.unique(labels.numpy())  # store unnormalized unique labels, used for phase indicator plot

        if normalize:
            self.y_min, self.y_max = labels.min().item(), labels.max().item()
            labels = normalize01(labels, self.y_min, self.y_max)
            self.samples = list(zip(inputs, labels))
        
        # used for phase indicator computation
        self.sorted_label_indices = np.argsort(labels)
        self.sorted_labels = labels.numpy()[self.sorted_label_indices] 
        self.unique_labels = np.unique(self.sorted_labels)
        print(f"unique labels: {self.unique_labels}")


    @property
    def discrete_labels(self):
        return self._discrete_labels
    
    def get_phase_indicator(self, out):
        """
        # TODO: address classification case.
        # TODO: remove redundancy in Paris_Data class & here.
        """

        out = out[:,0].cpu().numpy()

        if self.normalize:
            out = denormalize01(out, self.y_min, self.y_max)

        avg_sorted_out = [          # sort out by label, and compute mean per label
            out[self.sorted_label_indices][self.sorted_labels == label].mean()
            for label in self.unique_labels
        ]

        # Denormalize unique labels if necessary
        temp = denormalize01(self.unique_labels, self.y_min, self.y_max) if self.normalize else self.unique_labels
        return np.gradient(avg_sorted_out, temp).tolist()
    
    @property
    def data(self):
        return torch.stack([x[0] for x in self.samples])
    
    @property
    def labels(self):
        return torch.tensor([x[1] for x in self.samples])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

