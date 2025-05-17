import torch 
import torch.nn as nn
from train import Trainer, LDMTrainer
from utils import get_arguments, count_parameters
from video_diffusion_pytorch import GaussianDiffusion, Unet3D

def set_model(args):
    unet = Unet3D(
        dim=32
    )
    
    diffusion = GaussianDiffusion(
        unet,
        image_size=64,
        args=args,
        beta_schedule = 'cosine',
        auto_normalize=False,  # whether to normalize the image
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

def generate(itertions=100):
    args = get_arguments()
    diffusion = set_model(args)
    count_parameters(diffusion)
    if args.latent:
        trainer = LDMTrainer(diffusion, args, dataloading=True)
    else:
        trainer = Trainer(diffusion, args, dataloading=True)
    trainer.load_model()
    
    # unconditioned generation
    trainer.generatemp4(latent=args.latent, usinggaussian=args.usinggaussian)

def main():
    train()
    generate(itertions=100)
    
if __name__ == '__main__':
    main()