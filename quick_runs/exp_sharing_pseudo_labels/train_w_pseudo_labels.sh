# This script uses the generated pseudo labels to train a new DQN model for experience sharing.
# Other users are encourages to update: LOG_FOLDER, IMAGES_FOLDER, PSEUDO_LABELS_FOLDER.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  TASK_SCAN=$1 # Type of scan to use for training (options are 'fat' and 'water') (view quick_runs/README.md for more)
  TASK_NUM=$2 # Localization task to use for training (view quick_runs/README.md for more)
  GPU=${3:-0}  # Which GPU to use (view quick_runs/README.md for more)
  MAX_EPOCHS=${4:-3} # The max number of epochs to train for (3 is recommended); valid values are integers >= 1
  LR=${5:-0.001} # Learning rate to use for training

  LOG_FOLDER="/raid/home/slai16/research_retrain/pseudo_labels/set2_task_${TASK_SCAN}_${TASK_NUM}"
  IMAGES_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"
  PSEUDO_LABELS_FOLDER="/raid/home/slai16/research_retrain/pseudo_labels/set2_task_${TASK_SCAN}_${TASK_NUM}/pseudo_labels_files"

  FILES="${IMAGES_FOLDER}/train_image_paths.txt ${PSEUDO_LABELS_FOLDER}/train_landmark_paths.txt"
  LOG_DIR=${LOG_FOLDER}/pseudo_label_training_logs

#  echo "GPU: ${GPU}"
#  echo "FILES: ${FILES}"
#  echo "LOG DIR: ${LOG_DIR}"
#  echo "LR: ${LR}"
#  echo "Max Epochs: ${MAX_EPOCHS}"

  python DQN.py \
    --task train  \
    --gpu ${GPU} \
    --files ${FILES} \
    --agents 1 \
    --logDir ${LOG_DIR} \
    --lr ${LR} \
    --max_epochs ${MAX_EPOCHS}

  exit
}