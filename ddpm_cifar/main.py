import torch 
import torch.nn as nn
from utils import get_arguments, count_parameters
from train import Trainer
from evaluating import img_sampler
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
        beta_schedule = 'cosine',
        auto_normalize=False,  # whether to normalize the image
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sigma = args.sigma,  # noise distribution
        objective =args.objective,  # loss function
    ) 
    
    return diffusion
    
def train():
    args = get_arguments()
    diffusion = set_model(args)
    trainer = Trainer(diffusion, args)
    print(trainer.diffusion.sigma)
    print(trainer.diffusion.objective)
    count_parameters(diffusion)
    trainer.train(args.epochs)

def generate():
    args = get_arguments()
    diffusion = set_model(args)
    trainer = Trainer(diffusion, args)
    epoch = trainer.load_model()
    sampler = img_sampler(trainer.diffusion, args)
    dataset = sampler.create_dataset(num_samples=20000//args.batchsize)
    sampler.save_images(dataset)
    print(f'Images saved to {args.objective}/{args.sigma}/')

def evaluate():
    args = get_arguments()
    sampler = img_sampler(None, args)
    sampler.calculate_fid()
    
if __name__ == '__main__':
    # train() train for training
    #generate()  # generate for generating    
    evaluate()  # evaluate for evaluating