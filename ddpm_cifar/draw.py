import torch  
import numpy as np  
import matplotlib.pyplot as plt

def imshow(img_tensor):  
    """  
    将tensor图像反归一化并转成numpy，方便imshow显示。  
    img_tensor: (C,H,W)  
    """  
    img = img_tensor.cpu().numpy()  
    img = np.clip(img, 0, 1)                # 限制区间0~1  
    img = np.transpose(img, (1, 2, 0))     # C,H,W -> H,W,C  
    return img  

def show_images(images,epoch,std):  
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
    plt.savefig(f'./images/epoch_{epoch}_std{std}.png')  # 保存图片