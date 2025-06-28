#!/bin/bash

set -e

# set python path according to your actual environment
pythonpath='python'

# processing data

# cd data/ConstructDataset
# ${pythonpath} construct_dataset.py --gpu 2
# ${pythonpath} emotion_extract.py
# cd ..
# rm dataset_preproc.p
# cd ConstructDataset
# ${pythonpath} construct_dataset.py
# cd ../../

#${pythonpath} main.py \
#           --dataset data/dataset_preproc.p \
#            --is_pretrain True \
#            --is_with_pretrain True \
#            --is_train True \
#            --is_test True \
#            --is_evaluate True \
#            --is_evaluate_coher_elicit True \
#            --save_evaluate_log save/log/evaluate.log \
#            --save_test_log save/log/test.log \
#            --turn_reward_weight 1 \
#            --conversation_reward_weight 0.1 \
#            --context_reward_weight 0.1 \
#            --future_reward_weight 1 \
#            --save_method rewards \
#            --results_file save/results/results.json \
#            --gpu 0

${pythonpath} main.py \
  --dataset data/dataset_preproc.p \
  --is_pretrain False \
  --is_with_pretrain True \
  --is_train True \
  --train_epochs 2 \
  --turn_reward_weight 0.5 \
  --context_reward_weight 0.3 \
  --future_reward_weight 1.2 \
  --conversation_reward_weight 1.0 \
  --gpu 0