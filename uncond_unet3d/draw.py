import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

def show_videos(videos, title=None, batchsize=8, frames=16, save=False, path=None):
    """
    Show videos in a grid format and frames in a row.

    Args:
        videos (torch.Tensor): shape (b, t, c, h, w), in range [-1, 1]
        title (str): title for the entire plot
        batchsize (int): number of videos to display from batch
        frames (int): number of frames (unused here but kept for API consistency)
        save (bool): whether to save the plot
        path (str): path to save the image
    """

    # Take subset of batch and move to CPU
    videos = videos[:batchsize].detach().cpu()

    # Normalize videos from [-1, 1] to [0, 1]
    videos = ((videos + 1) / 2).clamp(0, 1)

    b, t, c, h, w = videos.shape

    # Reshape videos to (b*h, t*w, c) numpy array for display
    videos = videos.numpy().transpose(0, 3, 1, 4, 2).reshape(b * h, t * w, c)

    # Create a white canvas to place hints and videos
    
    # save the videos
    plt.imshow(videos)
    plt.axis('off')
    
    # Set overall title if provided  
    if title:  
        plt.title(title)  

    # Save figure if requested  
    if save and path is not None:  
        plt.savefig(path, bbox_inches='tight', pad_inches=0)  

    plt.show()  
    plt.close()  

def images2video(videos, path):
    """
    Save videos as mp4 files.
    Args:
        videos (torch.Tensor): Input videos of shape (batchsize, frames, channels, height, width) in range [-1, 1]
        path (str): Path to save the video files.
    """
    b, f, c, h, w = videos.shape
    files = os.listdir(path)
    files = [file for file in files if file.endswith('.mp4')]
    num = len(files) # used for naming the video
    for i in range(b):
        video = videos[i].permute(0,2,3,1).cpu().numpy()
        video = (video/2+0.5).clip(0, 1)
        video = (video * 255).astype(np.uint8) 
        # clip = ImageSequenceClip(list(video), fps=5)
        # clip.write_videofile(f'{path}/video_{num+i}.mp4')
        imageio.mimwrite(f'{path}/video_{num+i}.mp4', video, 
                        fps=5, quality=9)
    print(f'{num+b} videos have been saved in {path}')
