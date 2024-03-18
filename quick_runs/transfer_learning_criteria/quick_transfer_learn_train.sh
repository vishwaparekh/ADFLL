# For a given localization task, this script trains 18 finetuned models (each one using a different pretrained model).
# Other users are encourages to update: TRAIN_LOGS, PARAM_FOLDER, TASK_FOLDER.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  MODE=$1 # Breaks the training into partitions that can be run on separate terminals; options are [0, 1]
  TASK_SCAN=$2 # Modality of the finetuned models (view quick_runs/README.md on TASK_SCAN for more)
  TASK_NUM=$3 # Localization task for the finetuned models (view quick_runs/README.md for more)
  EPOCH=$4 # Number of epochs to train for each model; valid options are integer >= 1
  GPU="${5:-0}" # The GPU device to use (view quick_runs/README.md for more)
  LR="${6:-0.0001}" # LR to use, defaults to 0.0001

  ITER=$(( $EPOCH * 25000 ))
  TRAIN_LOGS="/raid/home/slai16/research_retrain/train_TL_LR_${LR}/transfer_learn_task_${TASK_SCAN}_${TASK_NUM}"
  PARAM_FOLDER="/raid/home/slai16/research_retrain/train_log_36_models"
  TASK_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"

  if [[ $MODE == 0 ]]; then
    echo "TRANSFER LEARNING FROM SET 1 FAT SCANS"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "STARTING TRANSFER LEARNING FROM set_1_fat_$counter"
      PARAM_FILE="$PARAM_FOLDER/set_1_fat_$counter/dev/model-${ITER}"
      FILES="$TASK_FOLDER/train_image_paths.txt $TASK_FOLDER/train_landmark_paths.txt"
      LOG_DIR=$TRAIN_LOGS/transfer_from_set_1_fat_$counter

#      echo "${PARAM_FILE}"
#      echo "${FILES}"
#      echo "${LOG_DIR}"
#      echo "${GPU}"

      python DQN.py \
        --task train  \
        --load ${PARAM_FILE} \
        --gpu ${GPU} \
        --files ${FILES} \
        --agents 1 \
        --logDir ${LOG_DIR} \
        --lr ${LR}
    done
  elif [[ $MODE == 1 ]]; then
    echo "TRANSFER LEARNING FROM SET 1 WATER SCANS"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "STARTING TRANSFER LEARNING FROM set_1_water_$counter"
      PARAM_FILE="$PARAM_FOLDER/set_1_wtr_$counter/dev/model-${ITER}"
      FILES="$TASK_FOLDER/train_image_paths.txt $TASK_FOLDER/train_landmark_paths.txt"
      LOG_DIR=$TRAIN_LOGS/transfer_from_set_1_water_$counter

#      echo "${PARAM_FILE}"
#      echo "${FILES}"
#      echo "${LOG_DIR}"
#      echo "${GPU}"

      python DQN.py \
        --task train  \
        --load ${PARAM_FILE} \
        --gpu ${GPU} \
        --files ${FILES} \
        --agents 1 \
        --logDir ${LOG_DIR} \
        --lr ${LR}
    done
  fi

  exit
}