#!/bin/bash

(nohup python -u main.py --batchsize 32 --objective pred_noise \
    --sigma 2.0 --device 4 > logfiles/generating/pred_noise/sd2.0.log 2>&1 )& \
    
(nohup python -u main.py --batchsize 32 --objective pred_noise \
    --sigma 1.5 --device 4 > logfiles/generating/pred_noise/sd1.5.log 2>&1 )& \

(nohup python -u main.py --batchsize 32 --objective pred_noise \
    --sigma 1.2 --device 3 > logfiles/generating/pred_noise/sd1.2.log 2>&1 )& 

(nohup python -u main.py --batchsize 32 --objective pred_noise \
    --sigma 1.0 --device 3 > logfiles/generating/pred_noise/sd1.0.log 2>&1 )& 

(nohup python -u main.py --batchsize 32 --objective pred_x0 \
    --sigma 2.0 --device 2 > logfiles/generating/pred_x0/sd2.0.log 2>&1 )& \
    
(nohup python -u main.py --batchsize 32 --objective pred_x0 \
    --sigma 1.5 --device 2 > logfiles/generating/pred_x0/sd1.5.log 2>&1 )& \

(nohup python -u main.py --batchsize 32 --objective pred_x0 \
    --sigma 1.2 --device 1 > logfiles/generating/pred_x0/sd1.2.log 2>&1 )& 

(nohup python -u main.py --batchsize 32 --objective pred_x0 \
    --sigma 1.0 --device 1 > logfiles/generating/pred_x0/sd1.0.log 2>&1 )& 