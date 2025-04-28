import os 
import sys
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import UCF
from einops import rearrange
from omegaconf import OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))  
# 添加 videodiffusion 到 Python 路径  
sys.path.insert(-1, os.path.join(current_dir, 'latent-diffusion/')) 
from ldm.models.autoencoder import AutoencoderKL, VQModelInterface 

class Trainer:
    def __init__(self, diffusion, args, dataloading=None):
        """initialize the trainer
        Args:
            diffusion: the diffusion model
            args: the arguments
            dataloading: whether to load the data or not 
        """
        self.args = args
        if dataloading:
            data = UCF(self.args)
            self.dataloader = data.load_data()
        self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
        self.diffusion = diffusion.to(self.device)
        self.optimizer = optim.AdamW(self.diffusion.parameters(), lr=args.lr)
        self.objective = args.objective
        
        # set the encoder 
        self.encoder_type = self.params.encoder 
        self.vae = self.create_vae()
        if self.encoder_type == 'autoencoderkl':
            self.scaling_factor = self.vae.config.scaling_factor
        elif self.encoder_type == 'vqgan':
            self.scaling_factor = 1.0
        else:
            raise ValueError('encoder type is not supported')
        
    def create_vae(self):
        if self.encoder_type == 'autoencoderkl':
            config_path =  "./stabilityai/sd-vae-ft-ema"
            vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path = 
                                                config_path).to(self.device)
        elif self.encoder_type == 'vqgan':
            config_path = "./latent-diffusion/models/first_stage_models/vq-f8/config.yaml"
            pathway = './latent-diffusion/models/first_stage_models/vq-f8/model.ckpt'
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
        """encode the input x to latent space
        Args:
            x: the input tensor of shape [b*t, c, h, w]
        Returns:
            z: the latent tensor of shape [b*t, c, h, w]
        """
        # x = rearrange(x, 'b t c h w -> (b t) c h w')
        z = self.vae.encode(x)
        if self.encoder_type == 'autoencoderkl':
            z = z.latent_dist.sample()*self.scaling_factor # shape [b*t, 4, h//8, w//8]
        elif self.encoder_type == 'vqgan':
            pass
        else:
            raise ValueError('encoder type is not supported')

        # for 2d u net z shaped [b*t, 4, h//8, w//8] here
        
        return z
    
    @torch.no_grad()
    def decode(self, z, batchsize) -> torch.Tensor:
        """decode the latent z to image space
        Args:
            z: the latent tensor of shape [b, t, c, h, w]
            batchsize: the batch size
        Returns:
            x: the image tensor of shape [b, t, c, h, w]
        """
        # z = rearrange(z, 'b c t h w -> b t c h w')
        # z = rearrange(z, 'b t c h w -> (b t) c h w')
        x = self.vae.decode(z/self.scaling_factor)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=batchsize)
        return x
   
    def train(self, epochs):
        for epoch in range(epochs):
            # using tqdm to show the progress bar
            for i, (images, _) in enumerate(self.dataloader):
                images = images.to(self.device)

                # forward pass
                loss = self.diffusion(images)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch {epoch}, Step [{i}/{len(self.dataloader)}], Loss: {loss.item():.4f}')
            
            # save the model
            self.save_model(epoch)
            self.validate(epoch)
    
    # @torch.no_grad()
    def save_model(self, epoch):
        """save the model and the optimizer state and epoch
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'./models/{self.objective}/model_{self.args.sigma}.pth')
        print(f'Model saved at epoch {epoch}')
    
    @torch.no_grad()
    def validate(self, epoch):
        """validate the model
        """
        self.diffusion.eval()