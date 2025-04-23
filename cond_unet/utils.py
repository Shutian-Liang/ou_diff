import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from prettytable import PrettyTable
# from torchvision.datasets import CIFAR10

def get_arguments():
    parser = argparse.ArgumentParser()
    # add parameters
    parser.add_argument('--sigma', type=float, default=1.0, help='noise distribution')
    parser.add_argument('--batchsize',type=int,default=64,help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=int, default=3, help='the device to use')
    parser.add_argument('--objective', type=str, default='pred_x0', help='object to train on')
    args = parser.parse_args()

    # get arguments
    return args