#!/bin/bash

# batch 2
# (nohup python -u evaluating.py --device 2 --training_noise ou \
#     --history 0 > logfiles/eval/ou_os_noseen.log 2>&1 )& \

# (nohup python -u evaluating.py --device 3 --training_noise gaussian \
#     --sampling_noise ou --history 1 > logfiles/eval/gaussian_os_seen.log 2>&1 )& \

# (nohup python -u evaluating.py --device 4 --training_noise gaussian \
#     --sampling_noise gaussian --history 1 > logfiles/eval/gaussian_gs_seen.log 2>&1 )& \

### here is the no seen here
# (nohup python -u evaluating.py --device 3 --training_noise gaussian \
#     --sampling_noise ou --history 0 > logfiles/eval/gaussian_os_unseen.log 2>&1 )& \

# (nohup python -u evaluating.py --device 2 --training_noise gaussian \
#     --sampling_noise gaussian --history 0 > logfiles/eval/gaussian_gs_unseen.log 2>&1 )& \


(nohup python -u evaluating.py --device 3 --training_noise ou \
    --sampling_noise gaussian --history 0 > logfiles/eval/ou_gs_unseen.log 2>&1 )& \

(nohup python -u evaluating.py --device 2 --training_noise ou \
    --sampling_noise gaussian --history 1 > logfiles/eval/ou_gs_seen.log 2>&1 )& \
