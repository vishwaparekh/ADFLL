# For a given finetuning localization task, this script evaluates 18 finetuned models (each one using a different pretrained model)
# across the 3 available checkpoints.
# Other users are encourages to update: LOG_FOLDER, IMAGES_FOLDER, PSEUDO_LABELS_FOLDER.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  # Run on 5 separate terminals; for one terminal, evaluate one transfer learn task for epochs 1, 2, and 3

  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  GPU=$1 # GPU device to use (view quick_runs/README.md for more)
  SCAN=$2 # The modality to evaluate (view quick_runs/README.md on TASK_SCAN for more)
  CLS=$3 # The task to evaluate (view quick_runs/README.md on TASK_NUM for more)

  for epochs in 1 2 3; do

#    echo "EPOCH: ${epochs}"
#    echo "SCAN: ${SCAN}"
#    echo "CLASS: ${CLS}"
#    echo "GPU: ${GPU}"

    bash quick_runs/transfer_learning_criteria/quick_transfer_learn_eval.sh 0 ${SCAN} ${CLS} ${epochs} ${GPU}
    bash quick_runs/transfer_learning_criteria/quick_transfer_learn_eval.sh 1 ${SCAN} ${CLS} ${epochs} ${GPU}
    bash quick_runs/transfer_learning_criteria/quick_transfer_learn_eval_baseline.sh ${SCAN} ${CLS} ${epochs} ${GPU}
  done

  exit
}