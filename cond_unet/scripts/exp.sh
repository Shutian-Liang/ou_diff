#!/bin/bash

### training next
# (nohup python -u main.py --batchsize 32 --num_samples 5 \
#     --device 4 --theta 1.579 --D 0.0384 --dt 0.05 --phi 1.5 > logfiles/pred_x0/ou1.5_dt005.log 2>&1 )& \

# (nohup python -u main.py --batchsize 32 --num_samples 5 \
#     --device 2 --theta 1.579 --D 0.0384 --dt 0.1 --phi 1.5 > logfiles/pred_x0/ou1.5_dt01.log 2>&1 )& \

# generating videos
# (nohup python -u main.py --batchsize 8 --num_samples 5 \
#     --device 3 --usingseed 1 --noise gaussian \
#     --usinggaussian 0 > logfiles/gen/gaussian_os.log 2>&1 )& \

# (nohup python -u main.py --batchsize 8 --num_samples 5 \
#     --device 4 --usingseed 1 --noise ou \
#     --usinggaussian 1 > logfiles/gen/ou_gs.log 2>&1 )& \

(nohup python -u main.py --batchsize 8 --num_samples 5 \
    --device 4 --usingseed 1 --noise ou --theta 1.579 --D 0.0384 --dt 0.05 --phi 1.5 \
    --usinggaussian 0 > logfiles/gen/ou1.579_os.log 2>&1 )& \

