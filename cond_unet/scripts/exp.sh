#!/bin/bash

(nohup python -u main.py --batchsize 32 --num_samples 5 \
    --device 4 > logfiles/pred_x0/ou1.log 2>&1 )& \

(nohup python -u main.py --batchsize 32 --num_samples 5 \
    --device 2 --noise gaussian > logfiles/pred_x0/gaussian.log 2>&1 )& \