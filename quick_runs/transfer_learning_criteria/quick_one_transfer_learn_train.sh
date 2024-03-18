# This script trains one transfer learned model (given a goal task, a pretrained task).
# Other users are encourages to update: TRAIN_LOGS, PARAM_FOLDER, TASK_FOLDER.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  # WARNING: the default learning rate is used in DQNModel.py

  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  TASK_SCAN=$1 # The modality to train the finetuned model on (view quick_runs/README.md on TASK_SCAN for more)
  TASK_NUM=$2 # The task to train the finetuned model on (view quick_runs/README.md on TASK_NUM for more)
  TL_SCAN=$3 # Modality the pretrained model is trained on
  TL_NUM=$4 # Task the pretrained model is trained on
  EPOCH=$5 # Number of epochs to train for; valid options are integers >= 1
  GPU="${6:-0}" # The GPU device to use (view quick_runs/README.md for more)

  ITER=$(( $EPOCH * 25000 ))
  TRAIN_LOGS="/raid/home/slai16/research_retrain/transfer_learn_reduce_lr_task_${TASK_SCAN}_${TASK_NUM}"
  PARAM_FOLDER="/raid/home/slai16/research_retrain/train_log_36_models"
  TASK_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"

  PARAM_FILE="$PARAM_FOLDER/set_1_${TL_SCAN}_${TL_NUM}/dev/model-${ITER}"
  FILES="$TASK_FOLDER/train_image_paths.txt $TASK_FOLDER/train_landmark_paths.txt"
  LOG_DIR=${TRAIN_LOGS}/transfer_from_set_1_${TL_SCAN}_${TL_NUM}

#  echo "${PARAM_FILE}"
#  echo "${FILES}"
#  echo "${LOG_DIR}"
#  echo "${GPU}"

  echo "TRANSFER LEARNING FROM set_1_${TL_SCAN}_${TL_NUM}"
  python DQN.py \
    --task train  \
    --load ${PARAM_FILE} \
    --gpu ${GPU} \
    --files ${FILES} \
    --agents 1 \
    --logDir ${LOG_DIR}

  exit
}