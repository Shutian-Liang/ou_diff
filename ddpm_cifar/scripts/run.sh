#!/bin/bash

(nohup python -u main.py --batchsize 32 --sigma 2.0 --device 4 > logfiles/sd2.0.log 2>&1 )& 
(nohup python -u main.py --batchsize 32 --sigma 1.5 --device 4 > logfiles/sd1.5.log 2>&1 )& 
(nohup python -u main.py --batchsize 32 --sigma 1.2 --device 3 > logfiles/sd1.2.log 2>&1 )& 
(nohup python -u main.py --batchsize 32 --sigma 1.0 --device 3 > logfiles/sd1.0.log 2>&1 )& 