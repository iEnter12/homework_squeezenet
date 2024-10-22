import argparse
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from SqueezeNet import MySqueezeNet
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

test_data = datasets.CIFAR10(
    root='./CIFAR10/',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, shuffle=True, batch_size=opt.batch_size)