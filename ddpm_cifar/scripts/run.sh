#!/bin/bash

# (nohup python -u main.py --batchsize 32 --objective pred_noise \
#     --sigma 0.5 --device 2 > logfiles/training/pred_noise/sd0.5.log 2>&1 )& \
    
# (nohup python -u main.py --batchsize 32 --objective pred_noise \
#     --sigma 0.3 --device 3 > logfiles/training/pred_noise/sd0.3.log 2>&1 )& \

# (nohup python -u main.py --batchsize 32 --objective pred_noise \
#     --sigma 1.2 --device 3 > logfiles/pred_noise/sd1.2.log 2>&1 )& 

# (nohup python -u main.py --batchsize 32 --objective pred_noise \
#     --sigma 1.0 --device 3 > logfiles/pred_noise/sd1.0.log 2>&1 )& 

# (nohup python -u main.py --batchsize 32 --objective pred_x0 \
#     --sigma 0.5 --device 2 > logfiles/training/pred_x0/sd0.5.log 2>&1 )& \
    
(nohup python -u main.py --batchsize 32 --objective pred_x0 \
    --sigma 0.3 --device 4 > logfiles/training/pred_x0/sd0.3.log 2>&1 )& \

# (nohup python -u main.py --batchsize 32 --objective pred_x0 \
#     --sigma 0.4 --device 4 > logfiles/training/pred_x0/sd0.4.log 2>&1 )& 

# (nohup python -u main.py --batchsize 32 --objective pred_x0 \
#     --sigma 1.0 --device 3 > logfiles/pred_x0/sd1.0.log 2>&1 )& 