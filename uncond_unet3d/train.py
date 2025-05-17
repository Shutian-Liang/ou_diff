import os 
import sys
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import UCF, checkandcreate
from einops import rearrange
from omegaconf import OmegaConf
from draw import show_videos, images2video 

current_dir = os.path.dirname(os.path.abspath(__file__))  
# 添加 videodiffusion 到 Python 路径  
sys.path.insert(-1, os.path.join(current_dir, 'latent-diffusion/')) 
from ldm.models.autoencoder import AutoencoderKL, VQModelInterface 


class Trainer:
    def __init__(self, diffusion, args, dataloading=None):
        """initialize the trainer on pixel level
        Args:
            diffusion: the diffusion model
            args: the arguments
            dataloading: whether to load the data or not 
        """
        self.args = args
        if dataloading:
            data = UCF(self.args)
            self.trainloader, self.testloader = data.load_data(usingseed=self.args.usingseed)
        self.testfiles = iter(self.testloader)
        self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
        self.diffusion = diffusion.to(self.device)
        self.optimizer = optim.AdamW(self.diffusion.parameters(), lr=args.lr)
        self.objective = args.objective
        self.noise = args.noise
        
        # for simple trainer dont use the vae encoder
        self.latent = False 
        
        
    def train(self, epochs):
        """train the diffusion model
        Args:
            epochs: the number of epochs to train
        """
        for epoch in range(epochs):
            # using tqdm to show the progress bar
            for i, (videos, _) in enumerate(self.trainloader):
                loss = self.forward(videos, latent=self.latent)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch {epoch}, Step [{i}/{len(self.trainloader)}], Loss: {loss.item():.4f}')
            
            # save the model
            self.save_model(epoch, self.latent)
            self.validate(epoch, self.latent)
            
    def forward(self, videos, latent=False):
        """forward pass of the model
        Args:
            videos: the input tensor of shape [b, f, c, h, w] -> [b, c, f, h, w]
            latent: whether to use the latent space or not
        Returns:
            loss: the loss value
        """
        videos = videos.to(self.device)
        videos = rearrange(videos, 'b f c h w -> b c f h w')
        if latent:
            videos = self.vae.encode(videos)
        loss = self.diffusion(videos)
        return loss
    
    def save_model(self, epoch, latent=False):
        """save the model and the optimizer state and epoch
        Args:
            epoch: the current epoch
        """
        enc = 'vae' if latent else 'pixel'
        path = f'./models/{self.objective}/{enc}/'
        checkandcreate(path)
        # torch.save({
        #     'epoch': epoch,
        #     'args': self.args,
        #     'model_state_dict': self.diffusion.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        # }, path+f'{self.noise}.pth')
        
        # compare different parameters
        torch.save({
            'epoch': epoch,
            'args': self.args,
            'model_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+f'{self.noise}_{self.args.theta}_{self.args.D}_{self.args.dt}_{self.args.phi}.pth')
        print(f'Model saved at epoch {epoch}')
    
    @torch.no_grad()
    def validate(self, epoch, latent=False, usinggaussian=False):
        """validate the model
        params:
            epoch: the current epoch
            latent: whether to use the latent space or not
        """
        self.diffusion.eval()
        
        # [b c f h w] -> [b f c h w]
        videos = self.diffusion.sample(batch_size=8, usinggaussian=usinggaussian)
        videos = rearrange(videos, 'b c f h w -> b f c h w')
        if latent:
            videos = self.vae.decode(videos,self.args.t)
            
        # files for saving the videos
        enc = 'vae' if self.latent else 'pixel'
        sn = 'gs' if usinggaussian else 'os'
        # path = f'./images/{self.objective}/{enc}/{self.noise}/{sn}/'
        
        # compare different parameters
        path = f'./images/{self.objective}/{enc}/{self.noise}_{self.args.theta}_{self.args.D}_{self.args.dt}_{self.args.phi}/{sn}/'
        
        checkandcreate(path)

        # save the videos
        show_videos(videos, frames=self.args.frames, title=self.noise, path=path+f'epoch{epoch}_seen.png', save=True)
        self.diffusion.train()
    
    def load_model(self):
        """load the model"""
        enc = 'vae' if self.latent else 'pixel'
        # checkpoint = torch.load(f'./models/{self.objective}/{enc}/{self.noise}.pth', weights_only=False, map_location=self.device)
        # compare different parameters
        checkpoint = torch.load(f'./models/{self.objective}/{enc}/{self.noise}_{self.args.theta}_{self.args.D}_{self.args.dt}_{self.args.phi}.pth', weights_only=False, map_location=self.device)
        
        self.diffusion.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Model loaded from epoch {epoch}',f'{self.objective}/{enc}/{self.noise}.pth')
        return epoch
    
    @torch.no_grad()
    def generatemp4(self, latent=False, usinggaussian=False):
        """generate mp4 videos from the hints
        Args:
            latent: whether to use the latent space or not
            usinggaussian: whether to use the gaussian noise or not
        """
        self.diffusion.eval()
        
        # [b c f h w] -> [b f c h w]
        videos = self.diffusion.sample(batch_size=8, usinggaussian=usinggaussian)
        videos = rearrange(videos, 'b c f h w -> b f c h w')
        
        # vae encode
        if latent:
            videos = self.vae.decode(videos,self.args.frames)
            
        # files for saving the videos
        enc = 'vae' if self.latent else 'pixel'
        sn = 'gs' if usinggaussian else 'os'
        # path = f'./videos/{self.objective}/{enc}/{self.noise}/{sn}/{history}/'
        # compare different parameters
        path = f'./videos/{self.objective}/{enc}/{self.noise}_{self.args.theta}_{self.args.D}_{self.args.dt}_{self.args.phi}/{sn}/'
        checkandcreate(path)
        images2video(videos, path=path)
    
class LDMTrainer(Trainer):
    def __init__(self, diffusion, args, dataloading=None):
        """initialize the ldm trainer
        Args:
            diffusion: the diffusion model
            args: the arguments
            dataloading: whether to load the data or not 
        """
        super().__init__(diffusion, args, dataloading)
        
        # set the encoder 
        self.latent = True
        self.encoder_type = self.args.encoder 
        self.vae = self.create_vae()
        
    def create_vae(self):
        if self.encoder_type == 'autoencoderkl':
            config_path =  "./stabilityai/sd-vae-ft-ema"
            vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path = 
                                                config_path).to(self.device)
            self.scaling_factor = vae.config.scaling_factor
        elif self.encoder_type == 'vqgan':
            config_path = "./latent-diffusion/models/first_stage_models/vq-f8/config.yaml"
            pathway = './latent-diffusion/models/first_stage_models/vq-f8/model.ckpt'
            config = OmegaConf.load(config_path)
            
            vae = VQModelInterface(**config.model.params).to(self.device)
            # load the model
            vae.load_state_dict(torch.load(pathway, map_location=self.device, weights_only=False)['state_dict'])
            self.scaling_factor = 1.0
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
        """encode the input x to latent space
        Args:
            x: the input tensor of shape [b*f, c, h, w]
        Returns:
            z: the latent tensor of shape [b*f, c, h, w]
        """
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        z = self.vae.encode(x)
        if self.encoder_type == 'autoencoderkl':
            z = z.latent_dist.sample()*self.scaling_factor # shape [b*t, 4, h//8, w//8]
        elif self.encoder_type == 'vqgan':
            pass
        else:
            raise ValueError('encoder type is not supported')
        
        # for unet 3d adding the batch size and shaped [b, t, c, h, w]
        z = rearrange(z, '(b f) c h w -> b f c h w', b=self.batchsize)
        return z
    
    @torch.no_grad()
    def decode(self, z, frames) -> torch.Tensor:
        """decode the latent z to image space
        Args:
            z: the latent tensor of shape [b, f, c, h, w]
            batchsize: the batch size
        Returns:
            x: the image tensor of shape [b, f, c, h, w]
        """
        # z = rearrange(z, 'b c t h w -> b t c h w')
        z = rearrange(z, 'b f c h w -> (b f) c h w')
        x = self.vae.decode(z/self.scaling_factor)
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frames)
        return x

