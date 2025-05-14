import argparse
from dataclasses import dataclass

@dataclass
class Params:
    lr: float  
    batchsize: int  
    epochs: int  
    frames: int
    workingpath: str
    noise: str  
    loss: str 
    theta: float
    D: float
    dt: float
    encoder: str
    sampling_method: str

def parse_args() -> Params:
    parser = argparse.ArgumentParser(description="Training parameters")
    
    # training patameters 
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')  
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training')  
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')  
    parser.add_argument('--frames', type=int, default=16, help='Number of frames in each video')
    parser.add_argument('--workingpath', type=str, default='./vdm', help='Path to save the model and videos')
    parser.add_argument('--noise', type=str, default='gaussian', choices=['gaussian','ou'], help='Type of noise to use')
    parser.add_argument('--loss', type=str, default='l2', choices=['l1', 'l2'], help='Type of loss to use')
    parser.add_argument('--encoder', type=str, default='vqgan', choices=['autoencoderkl','vqgan'], help='Encoder to use')
    
    # ou noise parameters
    parser.add_argument('--theta', type=float, default=1.0, help='Theta parameter for OU noise')
    parser.add_argument('--D', type=float, default=1.0, help='Sigma^2/2 parameter for OU noise')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step for OU noise')
    
    # sampling method for p_sample
    parser.add_argument('--sampling_method', type=str, default='sequence', 
                        choices=['sequence', 'independent'], help='Sampling method for videos generation')
    
    args = parser.parse_args()  
    return Params(**vars(args))