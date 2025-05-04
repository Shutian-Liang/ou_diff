import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange

def show_videos(videos, title=None, batchsize=8, frames=16, save=False, path =None):
    """
    Show videos in a grid format.
    Args:
        videos (torch.Tensor): Input videos of shape (batchsize, frames, channels, height, width) in range [-1, 1]
        title (str): Title for the plot.
        batchsize (int): Number of videos to display.
        frames (int): Number of frames in each video.
    """
    
    videos = videos[:batchsize].detach().cpu()
    videos = (videos + 1) / 2  # Normalize to [0, 1]
    videos = videos.clamp(0, 1)  # Clamp to [0, 1]
    b, t, c, h, w = videos.size()
    videos = videos.numpy().transpose(0, 3, 1, 4, 2).reshape(b*h, t*w, c)
    
    # show the formal videos
    plt.imshow(videos)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()  
    plt.show()
    if save:
        plt.savefig(path)
    plt.close()
    