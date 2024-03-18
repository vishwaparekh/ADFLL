# For a given finetuning localization task and an epoch checkpoint,
# evaluate all 18 finetuned models (each one using a different pretrained model) across different learning rates.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  # Run on 5 separate terminals;

  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  GPU=$1 # The GPU divice to use (view quick_runs/README.md for more)
  SCAN=$2 # The modality to evaluate (view quick_runs/README.md on TASK_SCAN for more)
  CLS=$3 # The task/class to evaluate (view quick_runs/README.md on TASK_NUM for more)
  EPOCH=$4 # The checkpoint to evaluate

  for lr in 0.0002 0.0004 0.0006 0.0008; do
    echo "----------------------------------------------------------------------"
    echo "Evaluating transfer learning (task ${SCAN} ${CLS}) with lr ${lr}"
    ROOT="/raid/home/slai16/research_retrain/Exp_TL_LR_${lr}"
    bash quick_runs/transfer_learning_criteria/quick_transfer_learn_eval.sh 0 ${SCAN} ${CLS} ${EPOCH} ${GPU} ${ROOT}
    bash quick_runs/transfer_learning_criteria/quick_transfer_learn_eval.sh 1 ${SCAN} ${CLS} ${EPOCH} ${GPU} ${ROOT}
  done

  exit
}