import os
import json  
from PIL import Image  
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset 
from pytorch_fid import fid_score
from tqdm import tqdm

class img_sampler:
    def __init__(self, diffusion, args):
        """sample the image from the model
        Args:
            model: the diffusion model
            args: the arguments
        """
        self.diffusion = diffusion
        self.args = args
        self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
        
        # set the transform
        self.transform = transforms.Compose([
            torchvision.transforms.Resize((299, 299)),  # FID需要299x299  
            torchvision.transforms.ToTensor()  
        ])
    
    def sample(self):
        self.diffusion.eval()
        images = self.diffusion.sample(self.args.batchsize)
        # images = F.interpolate(images, size=(299, 299))  # resize to 32x32
        return images  # 转成numpy数组，方便保存和显示
    
    def create_dataset(self, num_samples=1000):
        """create the dataset
        Args:
            num_samples: the number of samples to generate
        return: the dataset
        """
        dataset = torch.tensor([]).to(self.device)
        for i in range(num_samples):
            print(f'Generating {i+1}/{num_samples} images')
            img = self.sample()
            dataset = torch.cat((dataset, img), dim=0)
        dataset = dataset.cpu()
        dataset = TensorDataset(dataset)
        return dataset
    
    def save_images(self, dataset):
        """save the images to the disk
        Args:
            dataset: the dataset to save
        return: None    
        """
        save_dir = f'./cifar10/evaluated/{self.args.objective}/{self.args.sigma}/'  
        dataloader = DataLoader(dataset, batch_size=self.args.batchsize, num_workers=1, pin_memory=True)
        for batch_idx, imgs in enumerate(tqdm(dataloader)): 
            imgs = (imgs[0] + 1)/2  # 反归一化到0~1
            for i, img in enumerate(imgs):  
                img = torch.clamp(img, 0, 1)
                pil_img = transforms.ToPILImage()(img)  
                pil_img = pil_img.resize((299,299), resample=Image.BICUBIC)
                pil_img.save(os.path.join(save_dir, f'{batch_idx*64 + i}.png'))
    
    def calculate_fid(self, dataset):
        """calculate the fid score
        Args:
            dataset: the dataset to calculate
        return: the fid score
        """
        # load the dataset
        real_image_folder = './cifar10/evaluated/cifar10/'
        generated_images_folder = f'./cifar10/evaluated/{self.objective}/{self.sigma}/'
        # calculate the fid score
        fid = fid_score.calculate_fid_given_paths([real_image_folder, generated_images_folder], 64, self.device, 2048)
        
        # save the fid score
        fid_score_path = f'./cifar10/evaresults/{self.objective}/{self.sigma}/fid_score.json'
        if not os.path.exists(os.path.dirname(fid_score_path)):
            os.makedirs(os.path.dirname(fid_score_path))
        with open(fid_score_path, 'w') as f:
            json.dump({'fid_score': fid}, f)
        print(f'FID: {fid}')
        print(f'FID score saved to {fid_score_path}')

