import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import create_dataloader
from draw import show_images

# set the trainer
class Trainer:
    def __init__(self, diffusion, args):
        """initialize the trainer
        Args:
            diffusion: the diffusion model
            args: the arguments
        """
        self.args = args
        self.dataloader = create_dataloader(args.batchsize)
        self.device = 'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
        self.diffusion = diffusion.to(self.device)
        self.optimizer = optim.AdamW(self.diffusion.parameters(), lr=args.lr)
        
    def train(self, epochs):
        for epoch in range(epochs):
            # using tqdm to show the progress bar
            for i, (images, _) in enumerate(self.dataloader):
                images = images.to(self.device)

                # forward pass
                loss = self.diffusion(images)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    print(f'Epoch {epoch}, Step [{i}/{len(self.dataloader)}], Loss: {loss.item():.4f}')
        
            # save the model
            self.save_model(epoch)
            self.validate(epoch)
    
    @torch.no_grad()
    def save_model(self, epoch):
        """save the model and the optimizer state and epoch
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f'./models/model.pth')
        print(f'Model saved at epoch {epoch}')
    
    @torch.no_grad()
    def validate(self, epoch):
        """validate the model
        """
        self.diffusion.eval()
        images = self.diffusion.sample(self.args.batchsize)    
        show_images(images, epoch, self.args.sigma)
        self.diffusion.train()
    
    