import re 
import json
import pandas as pd

def get_eva_results(path, training_noise, sampling_noise, history, theta=1.0, dt = 0.1, usingpara = False):
    """
    get the evaluation results from the json files
    :param path: path to the json files
    :param training_noise: training noise
    :param sampling_noise: sampling noise(os or gs)
    :param history: history (seen or unseen)
    :return: evaluation results
    """
    
    def get_json_files(path, training_noise, sampling_noise, history):
        """
        get the json files from the path
        :param path: path to the json files
        :param training_noise: training noise
        :param sampling_noise: sampling noise(os or gs)
        :param history: history (seen or unseen)
        :return: json files
        """
        # get the json files
        if not usingpara:
            file = f'{path}/{training_noise}_{sampling_noise}_{history}_eval_results.json'
        else:
            file = f'{path}/{training_noise}_{sampling_noise}_{history}_{theta}_{dt}_eval_results.json'
        json_files = json.load(open(file))
        
        return json_files

    # define the dimension list
    dimension_list = ['subject_consistency', 'background_consistency', 
                      'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 
                      'aesthetic_quality', 'imaging_quality']
    
    # get the json files
    file = get_json_files(path, training_noise, sampling_noise, history)
    results = []
    label = []
    for dimension in dimension_list:
        subset = file[dimension][1]
        nums = len(subset)
        for num in range(nums):
            # per videos
            results.append(subset[num]['video_results'])
            label.append(dimension)
            
    # create the dataframe
    df = pd.DataFrame(results,columns=['video_results'])
    df['label'] = label
    df['training_noise'] = training_noise
    df['sampling_noise'] = sampling_noise
    df['history'] = history
    return df
    
    

