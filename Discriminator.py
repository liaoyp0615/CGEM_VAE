# Network used for the discriminator shown in the SI.
# Code written by Jordan
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np

class Flatten(nn.Module):
    '''
    Helper function to flatten a tensor.
    '''
    def forward(self, input):
            return input.view(input.size(0), -1)

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.Network = nn.Sequential(
            nn.Conv3d(1, 12, kernel_size=4, stride=1),
            nn.BatchNorm3d(12),
            nn.LeakyReLU(),
            nn.Conv3d(12, 24, kernel_size=4, stride=1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(),
            nn.Conv3d(24, 36, kernel_size=4, stride=2),
            nn.BatchNorm3d(36),
            nn.LeakyReLU(),
            nn.Conv3d(36, 48, kernel_size=4, stride=1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(),
            nn.Conv3d(48, 60, kernel_size=5, stride=2),
            nn.BatchNorm3d(60),
            nn.LeakyReLU(),
            nn.Conv3d(60, 72, kernel_size=4, stride=1),
            nn.BatchNorm3d(72),
            nn.LeakyReLU(),
            nn.Conv3d(72, 84, kernel_size=5, stride=2),
            nn.BatchNorm3d(84),
            nn.LeakyReLU(),
            nn.Conv3d(84, 96, kernel_size=4, stride=1),
            nn.BatchNorm3d(96),
            nn.LeakyReLU(),
            nn.Conv3d(96, 128, kernel_size=3, stride=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.Network(x)
        return x
