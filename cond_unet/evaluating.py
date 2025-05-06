import argparse
from vbench import VBench

def get_arguments():
    """
    get the arguments from the command line
    """
    parser = argparse.ArgumentParser()
    # noise setting
    parser.add_argument('--training_noise', type=str, default='ou', choices=['ou','gaussian'], help='Type of noise to use')
    parser.add_argument('--sampling_noise', type=str, default='ou', choices=['ou','gaussian'], help='Type of noise to use')
    
    # loading parameters
    parser.add_argument('--device', type=int, default=3, help='the device to use')
    parser.add_argument('--objective', type=str, default='pred_x0', help='object to train on')
    parser.add_argument('--latent', type=int, default=0, choices=[0,1], help='Use latent space or not')
    parser.add_argument('--history', type=int, default=0, choices=[0,1], help='Use history or not')
    args = parser.parse_args()
    
    # get arguments
    return args

class evaluator:
    def __init__(self, args, mode='custom_input'):
        self.mode = mode
        self.args = args
        self.training_noise = self.args.training_noise
        self.sampling_noise = self.args.sampling_noise
        self.dimension_list = ['subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
        self.device = self.args.device
        self.jsonpath = "./evaluation/VBench_full_info.json"
        self.savepath = "./evaluation/evaluation_results"
        self.vbench = VBench(self.device, self.jsonpath, self.savepath)
        self.videopath, self.samping, self.history = self.getvideopath()
        
    def getvideopath(self):
        """
        get the video path from the self.args
        """
        enc = 'vae' if self.args.latent else "pixel"
        sampling = 'gs' if self.sampling_noise == 'gaussian' else 'os'
        history = 'seen' if self.args.history else 'unseen'
        videopath = f'./videos/{self.args.objective}/{enc}/{self.training_noise}/{sampling}/{history}'
        return videopath, sampling, history
    
    def evaluating(self):
        """
        evaluate the videos
        """
        self.vbench.evaluate(videos_path = self.videopath, 
                             name = f'{self.training_noise}_{self.samping}_{self.history}', 
                             mode=self.mode,
                             dimension_list=self.dimension_list)
        print("Evaluation finished.")

if __name__ == "__main__":
    args = get_arguments()
    evaluator = evaluator(args)
    evaluator.evaluating()