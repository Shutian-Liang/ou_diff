import os
import argparse
import random
import shutil
import torch
from torchvision.datasets import UCF101
from torchvision.transforms import Compose, Lambda, Resize, Normalize
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from einops import rearrange
 
# from torchvision.datasets import CIFAR10

def get_arguments():
    """
    get the arguments from the command line
    """
    parser = argparse.ArgumentParser()
    # ou noise parameters
    parser.add_argument('--noise', type=str, default='ou', choices=['ou','gaussian'], help='Type of noise to use')
    parser.add_argument('--theta', type=float, default=1.0, help='Theta parameter for OU noise')
    parser.add_argument('--D', type=float, default=1.0, help='Sigma^2/2 parameter for OU noise')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step for OU noise')
    parser.add_argument('--usinggaussian', type=int, default=0, choices=[0,1], help='Use Gaussian noise or not')
    parser.add_argument('--phi', type=float, default=1.0, help='Standard deviation for initial noise')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batchsize',type=int,default=64,help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=int, default=3, help='the device to use')
    parser.add_argument('--objective', type=str, default='pred_x0', help='object to train on')
    parser.add_argument('--encoder', type=str, default='vqgan', choices=['autoencoderkl','vqgan'], help='Encoder to use')
    parser.add_argument('--latent', type=int, default=0, choices=[0,1], help='Use latent space or not')
    
    # dataset parameters
    parser.add_argument('--frames', type=int, default=16, help='Number of frames')
    parser.add_argument('--step_between_clips', type=int, default=1, help='Number of steps between clips')
    parser.add_argument('--side_size', type=int, default=64, help='Size of the side of the image')
    parser.add_argument('--subset', type=int, default=1, choices=[0,1], help='Use subset of the dataset')
    parser.add_argument('--usingseed', type=int, default=0, choices=[0,1], help='Use random seed or not')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for sampling')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to sample')
    parser.add_argument('--pictures', type=int, default=0, choices=[0,1], help='Use pictures of the dataset')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    # get arguments
    return args

class UCF():
    def __init__(self, args, path='../cond_unet/UCF101/'):
        self.args = args
        self.path = path
        self.frames = self.args.frames
        self.step_between_clips = self.args.step_between_clips
        self.side_size = self.args.side_size
        self.subset = self.args.subset
        self.random_seed = self.args.random_seed
        self.num_samples = self.args.num_samples
        self.pictures = self.args.pictures
        self.workers = self.args.workers
        self.batchsize = self.args.batchsize
        self.transforms = Compose([
            Lambda(lambda x: x.permute(0, 3, 1, 2)),  # THWC -> TCHW
            Lambda(lambda x: x / 255.0),
            # CenterCrop(356),
            Resize((self.side_size, self.side_size)),
            Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])
        self.train_data, self.test_data = self.get_data()
        self.train_data_len = len(self.train_data)
        
    def make_subset(self, train=True):
        """
        make a subset of the dataset
        params:
            train: whether to make a subset of the training set or not
        """
        random.seed(self.random_seed)

        # 设置路径  
        source_path = '../cond_unet/UCF101/UCF-101/'  
        datatype = 'train' if train else 'test'
        
        if not self.pictures:
            target_path = '../cond_unet/UCF101/UCF-101_' + datatype + '/'
        else:
            target_path = '../cond_unet/UCF101/UCF-101_pictures_' + datatype + '/'

        # 获取所有类别文件夹  
        folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]  
        folders = sorted(folders)  

        #每次抽样前清空目标文件夹  
        for folder in folders:  
            target_folder = os.path.join(target_path, folder)  
            os.makedirs(target_folder, exist_ok=True)  # 创建目标子文件夹  
            
            # # 清空目标文件夹  
            # for file in os.listdir(target_folder):  
            #     os.remove(os.path.join(target_folder, file))  

        #抽样并复制文件  
        sample_num = self.num_samples
        with open(f'../cond_unet/UCF101/ucfTrainTestlist/{datatype}list01.txt') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(' ')[0] for line in lines]
        lines = [line.split('/')[1] for line in lines]
        lines = [line.split('.')[0] for line in lines]
        lines = sorted(lines)
        lines = list(set(lines))
        lines = [line+'.avi' for line in lines]
        for folder in folders:  
            files = os.listdir(os.path.join(source_path, folder))  
            files = [file for file in files if file in lines]
            sample_files = random.sample(files, min(sample_num, len(files)))  # 确保不超过实际文件数量
            for file in sample_files:  
                shutil.copy(os.path.join(source_path, folder, file), os.path.join(target_path, folder, file))  
            
        print("Sampling completed successfully!")
        
    def get_data(self):
        if not self.subset:
            # full dataset
            train_root = self.path + 'UCF-101/'
        else:
            # subset
            self.make_subset()
            self.make_subset(train=False)
            if self.pictures:
                train_root = self.path + 'UCF-101_pictures_train/'
            else:
                train_root = self.path + 'UCF-101_train/'
                test_root = self.path + 'UCF-101_test/'   
        train_data = UCF101(
            root=train_root,  # subset数据集路径
            annotation_path=self.path + 'ucfTrainTestlist',  # 注释文件路径
            frames_per_clip=self.frames,  # 提取每个 clip 的帧数
            step_between_clips=self.step_between_clips,  # 每帧的间隔
            train=True,  # 加载训练集 (False 则加载测试集)
            transform=self.transforms,  # 帧级变换
            num_workers=4
        )
        
        test_data = UCF101(
            root=test_root,  # subset数据集路径
            annotation_path=self.path + 'ucfTrainTestlist',  # 注释文件路径
            frames_per_clip=1,  # as the hints
            step_between_clips=self.step_between_clips,  # 每帧的间隔
            train=False,  # 加载训练集 (False 则加载测试集)
            transform=self.transforms,  # 帧级变换
            num_workers=4
        )
        return train_data, test_data          
        
    def load_data(self, usingseed=0):
        def custom_collate(batchsize):
            filtered_batch = []
            for video, _, label in batchsize:
                filtered_batch.append((video, label))
            return torch.utils.data.dataloader.default_collate(filtered_batch)
        # no random seed
        # Reset random seed to None for shuffling
        random.seed(None)
        torch.manual_seed(torch.initial_seed())
        if usingseed:
            # set random seed
            g = torch.Generator()
            g.manual_seed(self.args.random_seed)
        else:
            g = None
        train_loader = DataLoader(self.train_data, batch_size=self.batchsize, shuffle=True, num_workers=4, pin_memory=True,
                                  collate_fn=custom_collate, drop_last=True, generator=g)
        test_loader = DataLoader(self.test_data, batch_size=self.batchsize, shuffle=True, num_workers=4, pin_memory=True,
                                  collate_fn=custom_collate, drop_last=True, generator=g)
        return train_loader, train_loader

    def reshape_input(self, videos):
        # raw shape [b,T,c,H,W]
        # now shape [b*T,c,H,W]
        videos = rearrange(videos, 'b t c h w -> (b t) c h w')
        return videos

def count_parameters(model):
    """Count the number of parameters in the model
    Args:
        model: the model to count parameters
    Returns:
        total_params: the total number of parameters in the model
    """
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

def checkandcreate(path):
    """
    check if the path exists, if not create it
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Created directory {path}')
    else:
        print(f'Directory {path} already exists')

class Namespace:  
    def __init__(self, **kwargs):  
        """
        use this class to create a namespace in jupyter notebook for debugging
        """
        self.__dict__.update(kwargs)   
