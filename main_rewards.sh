#!/bin/bash

set -e

pythonpath='python'

${pythonpath} main_rewards.py --direction forward --is_train True --is_test True --learning_rate 2e-5 --gpu 0

${pythonpath} main_rewards.py --direction backward --is_train True --is_test True --learning_rate 2e-5 --gpu 0
