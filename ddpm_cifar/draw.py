import re
import torch  
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def imshow(img_tensor):  
    """  
    将tensor图像反归一化并转成numpy，方便imshow显示。  
    img_tensor: (C,H,W)  
    """  
    img = img_tensor.cpu().numpy() 
    img = img * 0.5 + 0.5            # 反归一化到0~1 
    img = np.clip(img, 0, 1)                # 限制区间0~1  
    img = np.transpose(img, (1, 2, 0))     # C,H,W -> H,W,C  
    return img  

def show_images(images, objective, epoch, std, save=True):  
    """  
    images: tensor形状 (32, C, H, W)  
    """  
    images = images[:32]  # 确保最多32张  
    fig, axes = plt.subplots(4, 8, figsize=(16,8))  
    axes = axes.flatten()  

    for i in range(len(images)):  
        img = imshow(images[i])  # tensor转numpy图像  
        axes[i].imshow(img)  
        axes[i].axis('off')  
    for j in range(len(images), 32):  
        axes[j].axis('off')  

    plt.tight_layout()  
    # plt.show()  
    if save:
        plt.savefig(f'./images/{objective}/epoch_{epoch}_std{std}.png')  # 保存图片

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
    palette = sns.color_palette("tab10", n_colors=8)
    for i in range(len(dfs)):
        sns.lineplot(data=dfs[i], x='epoch', y='loss', hue='std', palette=palette, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend(title='Std', loc='upper right')
    plt.tight_layout()
    plt.show()