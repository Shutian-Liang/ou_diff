#!/bin/bash

(nohup python -u main.py --batchsize 32 \
    --device 4 > logfiles/pred_x0/ou1.log 2>&1 )& \