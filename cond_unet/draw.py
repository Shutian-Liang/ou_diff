import re
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from einops import rearrange

def show_videos(videos, hints, title=None, batchsize=8, frames=16, save=False, path=None, gap=20):  
    """  
    Show videos in a grid format with hints shown on the left side of each video,  
    and titles above hints and videos.  

    Args:  
        videos (torch.Tensor): shape (b, t, c, h, w), in range [-1, 1]  
        hints (torch.Tensor): shape (b, t, c, h, w), each batch frame is the same  
        title (str): title for the entire plot  
        batchsize (int): number of videos to display from batch  
        frames (int): number of frames (unused here but kept for API consistency)  
        save (bool): whether to save the plot  
        path (str): path to save the image  
        gap (int): horizontal gap in pixels between hints and videos  
    """  
    import torch  

    # Take subset of batch and move to CPU  
    videos = videos[:batchsize].detach().cpu()  
    hints = hints[:batchsize].detach().cpu()  

    # Normalize videos and hints from [-1, 1] to [0, 1]  
    videos = (videos + 1) / 2  
    videos = videos.clamp(0, 1)  

    hints = (hints + 1) / 2  
    hints = hints.clamp(0, 1)  

    b, t, c, h, w = videos.shape  

    # Extract the first frame of hints for each batch item: shape (b, c, h, w)  
    hint_imgs = hints[:, 0]  

    # Convert hints to numpy array and permute to (b, h, w, c)  
    hint_imgs = hint_imgs.permute(0, 2, 3, 1).numpy()  

    # If grayscale (c=1), repeat to make 3 channels for visualization  
    if c == 1:  
        hint_imgs = np.repeat(hint_imgs, 3, axis=3)  
        videos = videos.repeat(1, 1, 3, 1, 1)  # repeat channels for videos as well  
        c = 3  

    # Reshape videos to (b*h, t*w, c) numpy array for display  
    videos_np = videos.numpy().transpose(0, 3, 1, 4, 2).reshape(b * h, t * w, c)  

    # Reserve space on top for titles  
    title_height = 30  # pixels reserved for the title bar  
    total_width = w + gap + t * w  
    total_height = title_height + b * h  

    # Create a white canvas to place hints and videos  
    canvas = np.ones((total_height, total_width, c), dtype=np.float32)  

    # Paste each hint and video on the canvas  
    for i in range(b):  
        start_h = title_height + i * h  
        # Place hint on the left side  
        canvas[start_h:start_h + h, 0:w, :] = hint_imgs[i]  
        # Place videos to the right of hints (with gap)  
        canvas[start_h:start_h + h, w + gap:w + gap + t * w, :] = videos_np[i * h:(i + 1) * h, :, :]  

    plt.figure(figsize=(12, 3 * batchsize))  
    plt.imshow(canvas)  
    plt.axis('off')  

    ax = plt.gca()  

    # Draw 'hints' text centered above hints  
    ax.text(w / 2, title_height / 2, 'hints', fontsize=14, fontweight='bold',  
            ha='center', va='center', color='black')  

    # Draw 'generation' text centered above videos  
    ax.text(w + gap + (t * w) / 2, title_height / 2, 'generation', fontsize=14, fontweight='bold',  
            ha='center', va='center', color='black')  

    # Set overall title if provided  
    if title:  
        plt.title(title)  

    plt.tight_layout()  

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
    
def get_loss(log_file):
    records = []
    
    # 修改后的正则表达式 (主要变化在分隔符和字段匹配)
    pattern = r"Epoch (\d+),\s*Step\s*\[\d+/\d+\],\s*Loss:\s*([0-9.]+)"
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))  # 现在只有两个捕获组
                
                records.append({
                    'epoch': epoch,
                    'loss': loss
                })
    
    return pd.DataFrame(records)

def plot_loss(dfs,titles):
    """
    draw the loss curve with different std
    
    :param df: a list of different DataFrame with columns ['epoch', 'loss', 'std']
    :param titles: a list of titles for each subplot
    
    :return: None
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    palette = sns.color_palette("tab10", n_colors=4)
    for i in range(len(dfs)):
        sns.lineplot(data=dfs[i], x='epoch', y='loss', hue='std', palette=palette, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend(title='Std', loc='upper right')
    plt.tight_layout()
    plt.show()
    