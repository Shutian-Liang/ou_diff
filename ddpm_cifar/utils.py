import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from prettytable import PrettyTable
# from torchvision.datasets import CIFAR10
class Namespace:  
    def __init__(self, **kwargs):  
        """
        use this class to create a namespace in jupyter notebook for debugging
        """
        self.__dict__.update(kwargs)    
        
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

def create_dataloader(batchsize):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10 = datasets.CIFAR10(root='./cifar10', train=True, download=False, transform=transform)
    
    # generate the dataloader
    dataloader = DataLoader(cifar10, batch_size=batchsize, shuffle=True, num_workers=4,pin_memory=True)
    return dataloader

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params