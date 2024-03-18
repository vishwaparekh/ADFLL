# This script evaluates one transfer learned model (given a goal task, a pretrained task, a checkpoint).
# Other users are encourages to update: PARAM_FOLDER, VAL_LOGS, TASK_FOLDER.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  TASK_SCAN=$1 # The modality to evaluate (view quick_runs/README.md on TASK_SCAN for more)
  TASK_NUM=$2 # The task to evaluate (view quick_runs/README.md on TASK_NUM for more)
  TL_SCAN=$3 # Modality the pretrained model is trained on
  TL_NUM=$4 # Task the pretrained model is trained on
  EPOCH=$5 # The checkpoint to use for the finetuned model
  GPU="${6:-0}"  # The GPU device to use (view quick_runs/README.md for more)

  ITER=$(( 75000 + $EPOCH * 25000 ))
  PARAM_FOLDER="/raid/home/slai16/research_retrain/transfer_learn_reduce_lr_task_${TASK_SCAN}_${TASK_NUM}"
  VAL_LOGS="/raid/home/slai16/research_retrain/eval_tl_reduce_lr_finetune_${EPOCH}_epochs/eval_tl_task_${TASK_SCAN}_${TASK_NUM}"
  TASK_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"


  FILES="$TASK_FOLDER/val_image_paths.txt $TASK_FOLDER/val_landmark_paths.txt"
  PARAM_FILE="$PARAM_FOLDER/transfer_from_set_1_${TL_SCAN}_${TL_NUM}/dev/model-${ITER}"
  LOG_FOLDER="$VAL_LOGS/transfer_from_set_1_${TL_SCAN}_${TL_NUM}"

#  echo "${PARAM_FILE}"
#  echo "${FILES}"
#  echo "${LOG_FOLDER}"
#  echo "${GPU}"

  echo "EVALUATING MODEL (TASK ${TASK_SCAN} ${TASK_NUM}) TRANSFER LEARNED FROM ${TL_SCAN} LANDMARK ${TL_NUM}"

  python DQN.py \
    --task eval  \
    --gpu ${GPU} \
    --load ${PARAM_FILE} \
    --files ${FILES} \
    --logDir ${LOG_FOLDER} \
    --saveGif \
    --agents 1

  exit
}