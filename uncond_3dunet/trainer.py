# add sys path
import sys
import os
sys.path.append('../')
# 获取当前脚本的绝对路径  
current_dir = os.path.dirname(os.path.abspath(__file__))  

# 获取父目录（包含当前文件夹和 videodiffusion 的目录）  
parent_dir = os.path.dirname(current_dir)  

# 添加 videodiffusion 到 Python 路径  
sys.path.insert(-1, os.path.join(parent_dir, 'videodiffusion/latent-diffusion/')) 

# necessary packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from einops import rearrange
import torch.nn.functional as F
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
import importlib
from ldm.models.autoencoder import AutoencoderKL, VQModelInterface 


class Trainer:
    def __init__(self, diffusion, params, data=None):
        self.diffusion = diffusion
        self.params = params

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.diffusion.to(self.device)
        self.encoder_type = self.params.encoder    
        self.vae = self.create_vae()
        print(self.diffusion.noise)
        
        # parameters setting
        self.batchsize = self.params.batchsize
        self.sub_frames = self.params.frames // 4
        self.lr = self.params.lr
        self.epochs = self.params.epochs
        
        # save path
        self.noise = self.params.noise
        self.workingpath = self.params.workingpath
        self.sampling_method = self.params.sampling_method
        self.videopath = self.workingpath + '/videos/'+ f'{self.noise}/{self.sampling_method}/'
        self.modelpath = self.workingpath + '/models/'+ self.noise + '/'
        self.losspath = self.workingpath + '/loss/' + self.noise + '/'
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=self.lr)

        if data is not None:
            self.dataloader = data.load_data(batchsize=self.batchsize) 
        
        # scaling factor
        if self.encoder_type == 'autoencoderkl':
            self.scaling_factor = self.vae.config.scaling_factor
        elif self.encoder_type == 'vqgan':
            self.scaling_factor = 1.0
        else:
            raise ValueError('encoder type is not supported')
        
    def create_vae(self):
        if self.encoder_type == 'autoencoderkl':
            config_path =  "./videodiffusion/stabilityai/sd-vae-ft-ema"
            vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path = 
                                                config_path).to(self.device)
        elif self.encoder_type == 'vqgan':
            config_path = "./videodiffusion/latent-diffusion/models/first_stage_models/vq-f8/config.yaml"
            pathway = './videodiffusion/latent-diffusion/models/first_stage_models/vq-f8/model.ckpt'
            config = OmegaConf.load(config_path)
            
            vae = VQModelInterface(**config.model.params).to(self.device)
            # load the model
            vae.load_state_dict(torch.load(pathway, map_location=self.device, weights_only=False)['state_dict'])
        else:
            print(self.encoder_type)
            raise ValueError('encoder type is not supported')
        
        # freeze the parameters and dont require gradients
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        
        return vae      
    
    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        z = self.vae.encode(x)
        if self.encoder_type == 'autoencoderkl':
            z = z.latent_dist.sample()*self.scaling_factor # shape [b*t, 4, h//8, w//8]
        elif self.encoder_type == 'vqgan':
            pass
        else:
            raise ValueError('encoder type is not supported')
        z = rearrange(z, '(b t) c h w -> b t c h w', b=self.batchsize)
        
        # reshape z from b 4*t c h w to 4*b t c h w
        z = z.view(-1, self.sub_frames, *z.shape[2:])
        
        return z
    
    @torch.no_grad()
    def decode(self, z, batchsize) -> torch.Tensor:
        z = rearrange(z, 'b c t h w -> b t c h w')
        z = rearrange(z, 'b t c h w -> (b t) c h w')
        x = self.vae.decode(z/self.scaling_factor)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=batchsize)
        return x
        
    def train(self, sample_batchsize = 8):
        epoch_loss = []
        training_step = 0
        traindatalen = len(self.dataloader)
        print(f'start training: total {traindatalen} batches')
        for epoch in range(self.epochs):
            self.diffusion.train()
            losses = []
            for images, _ in self.dataloader:
                input = images.to(self.device)
                input = self.encode(input)
                input = rearrange(input, 'b t c h w -> b c t h w')
                loss = self.diffusion(input)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                training_step += 1
                
                # print(training_step)
                if training_step % 10 == 0:
                    print(f'epoch {epoch}; progress: {training_step  / traindatalen}; loss: {loss.item()/self.batchsize}')
                    losses.append(loss.item())

            sampled_videos = self.validate(sample_batchsize, self.sampling_method)
            self.save(epoch, sampled_videos)
            self.save_model(epoch)
            self.save_loss(epoch_loss)
            epoch_loss.append(sum(losses) / len(losses) / self.batchsize)
    
    @torch.no_grad()
    def validate(self, batchsize, sampling_method):
        self.diffusion.eval()
        
        # shape [b, c, t, h, w]
        sampled_videos = self.diffusion.sample(batch_size=batchsize,
                                               sampling_method=sampling_method)
        
        # [b, c, t, h, w]
        sampled_videos = self.decode(sampled_videos, batchsize)
        
        return sampled_videos
    
    def save(self, epoch, sampled_videos):
        videos = (sampled_videos/2+0.5).clamp(0, 1)
        b, t, c, h, w = videos.size()
        videos = videos.cpu().detach().numpy().transpose(0, 3, 1, 4, 2).reshape(b*h, t*w, c)
        
        # save the videos
        plt.imshow(videos)
        plt.axis('off')
        plt.title(f'{self.noise}')
        plt.savefig(self.videopath + f'{self.noise}_epoch{epoch}_{self.sampling_method}.png',
                    bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def save_model(self, epoch):
        savepath = self.modelpath + f'{self.noise}.pth'
        
        # save the epoch and optimizer and model state
        state = {
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'model': self.diffusion.state_dict(),
        }
        # save the model
        torch.save(state, savepath)
        print(f'model saved at epoch {epoch}')
    
    def save_loss(self, loss):
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=range(len(loss)), y=loss)
        plt.xticks(range(len(loss)))
        plt.ylabel('MSE loss')
        plt.savefig(self.losspath + 'loss.png')
        plt.close()


        