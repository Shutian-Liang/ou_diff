from utils import parse_args
from trainer import Trainer
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# add sys path
import sys
import os
import torch
sys.path.append('../')

#experiment configuration
import torch
from ddpm import count_parameters
from config import load_config

def main():
    params = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=4
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        channels=4,
        image_size = 8,
        num_frames = params.frames,
        timesteps = 1000,   # number of steps
        loss_type = params.loss,# L1 or L2
        theta = params.theta,
        noise = params.noise,
        D = params.D,
        dt = params.dt
    )
    print(params)
    count_parameters(diffusion)
    os.chdir('../')
    data, param, model_path = load_config(setting=2, 
                                          subset=True, num_samples=5,
                                          frames=params.frames)

    trainer = Trainer(diffusion=diffusion, params=params, data=data)
    trainer.train()

if __name__ == '__main__':
    main()
    