# File for repeating lattices.
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
from torch.utils.data.dataset import Dataset

class Flatten(nn.Module):
    '''
    Helper function to flatten a tensor.
    '''
    def forward(self, input):
            return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    # Convert to 3d matrices
    def forward(self, input):
            return input.view(input.size(0), 128, 1, 1, 1)

class Interpolate(nn.Module):
    '''
    Interpolate for upsampling. Use convolution and upsampling
    in favor of conv transpose.
    '''
    def __init__(self, scale_factor, mode):
            super(Interpolate, self).__init__()
            self.interp = nn.functional.interpolate
            self.scale_factor = scale_factor
            self.mode = mode

    def forward(self, x):
            x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
            return x


class CVAE(nn.Module):
    '''
    CGEM VAE.
    '''
    def __init__(self, input_channels=1, h_dim=128*(1*1*1), h_dim3=128*(1*1*1), h_dim2=3000, z_dim=300):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=6, stride=2), #92-44
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 32, kernel_size=5, stride=3), #44-14
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=5, stride=3), #14-4
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=1), #4-1
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            Flatten()
        )

        self.fc0 = nn.Linear(h_dim, 3000)
        self.fc1 = nn.Linear(3000, z_dim) # for mu
        self.fc2 = nn.Linear(3000, z_dim) # for logvar
        self.fc3 = nn.Linear(z_dim, h_dim2)
        self.fc4 = nn.Linear(h_dim2, h_dim3)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=1), #4
            nn.BatchNorm3d(64, 1e-3),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=5, stride=3), #14
            nn.BatchNorm3d(32, 1e-3),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=5, stride=3), #44
            nn.BatchNorm3d(16, 1e-3),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=6, stride=2), #92
            nn.BatchNorm3d(1, 1e-3),
            nn.ReLU()
        )

            

    def reparameterization(self, mu, logvar):
        # Reparamterization trick to backpropograte through the drawing of mu and sigma
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=device)
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        # Go through the small latent space.
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterization(mu, logvar)
        return z, mu, logvar


    def encode(self, x):
        # Go through the encoder.
        # *Input* is the original grid
        # *Output* is a vector to pass through the bottleneck.
        h = self.encoder(x)
        h =  self.fc0(h)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
