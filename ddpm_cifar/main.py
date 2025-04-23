import torch 
import torch.nn as nn
from utils import get_arguments, count_parameters
from train import Trainer
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def set_model(args):
    unet = Unet(
        dim=32,
        dim_mults=(1, 2, 4, 8),
        channels=3
    )
    
    diffusion = GaussianDiffusion(
        unet,
        image_size=32,
        timesteps=1000,  # number of steps
        sampling_timesteps=1000,
        auto_normalize=False,  # whether to normalize the image
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sigma = args.sigma,  # noise distribution
    ) 
    
    return diffusion
    
def main():
    args = get_arguments()
    diffusion = set_model(args)
    trainer = Trainer(diffusion, args)
    print(trainer.diffusion.sigma)
    count_parameters(diffusion)
    trainer.train(args.epochs)

if __name__ == '__main__':
    main()
    