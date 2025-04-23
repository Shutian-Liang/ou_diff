import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10

def get_arguments():
    parser = argparse.ArgumentParser()
    # add parameters
    parser.add_argument('--sigma', type=float, default=1.0, help='noise distribution')
    parser.add_argument('--batchsize',type=int,default=64,help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=int, default=3, help='the device to use')
    args = parser.parse_args()

    # get arguments
    return args

def create_dataloader(batchsize):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    cifar10 = datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
    
    # generate the dataloader
    dataloader = DataLoader(cifar10, batch_size=batchsize, shuffle=True, num_workers=4)
    return dataloader