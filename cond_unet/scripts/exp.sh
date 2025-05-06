#!/bin/bash

### training next
# (nohup python -u main.py --batchsize 32 --num_samples 5 \
#     --device 4 > logfiles/pred_x0/ou1.log 2>&1 )& \

# (nohup python -u main.py --batchsize 32 --num_samples 5 \
#     --device 2 --noise gaussian > logfiles/pred_x0/gaussian.log 2>&1 )& \

# generating videos
# (nohup python -u main.py --batchsize 8 --num_samples 5 \
#     --device 3 --usingseed 1 --noise gaussian \
#     --usinggaussian 0 > logfiles/gen/gaussian_os.log 2>&1 )& \

(nohup python -u main.py --batchsize 8 --num_samples 5 \
    --device 4 --usingseed 1 --noise gaussian \
    --usinggaussian 1 > logfiles/gen/gaussian_gs.log 2>&1 )& \

# (nohup python -u main.py --batchsize 8 --num_samples 5 \
#     --device 4 --usingseed 1 --noise ou \
#     --usinggaussian 0 > logfiles/gen/ou_os.log 2>&1 )& \

