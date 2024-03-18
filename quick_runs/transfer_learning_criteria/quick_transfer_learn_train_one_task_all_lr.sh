# Runs quick_transfer_learn_train.sh for various learning rates;
# this script transfer learns 18 models for one tasks across different LRs
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  # Run on 5 separate terminals (one terminal per task)

  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  TASK_SCAN=$1 # Modality of the finetuned models (view quick_runs/README.md on TASK_SCAN for more)
  TASK_NUM=$2 # Localization task for the finetuned models (view quick_runs/README.md for more)
  GPU="${3:-0}" # The GPU device to use (view quick_runs/README.md for more)

  for lr in 0.0002 0.0004 0.0006 0.0008; do
    echo "Transfer learning for task ${TASK_SCAN} ${TASK_NUM} on GPU ${GPU}; LR is ${lr}"

    bash quick_runs/transfer_learning_criteria/quick_transfer_learn_train.sh 0 ${TASK_SCAN} ${TASK_NUM} 3 ${GPU} ${lr}
    bash quick_runs/transfer_learning_criteria/quick_transfer_learn_train.sh 1 ${TASK_SCAN} ${TASK_NUM} 3 ${GPU} ${lr}
  done

  exit
}