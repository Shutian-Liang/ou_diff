import torch 
import torch.nn as nn
from train import Trainer, LDMTrainer
from utils import get_arguments, count_parameters
from diffusion import GaussianDiffusion, CondUnet

def set_model(args):
    unet = CondUnet(
        dim=32,
        size=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    )
    
    diffusion = GaussianDiffusion(
        unet,
        args=args,
        image_size=64,
        timesteps=1000,  # number of steps
        sampling_timesteps=1000,
        beta_schedule = 'cosine',
        auto_normalize=False,  # whether to normalize the image
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])  # noise distribution
        objective =args.objective,  # loss function
    ) 
    
    return diffusion

def train():
    args = get_arguments()
    diffusion = set_model(args)
    count_parameters(diffusion)
    if args.latent:
        trainer = LDMTrainer(diffusion, args, dataloading=True)
    else:
        trainer = Trainer(diffusion, args, dataloading=True)
    trainer.train(args.epochs)

if __name__ == '__main__':
    # train() train for training
    #generate()  # generate for generating    
    # evaluate()  # evaluate for evaluating
    train()