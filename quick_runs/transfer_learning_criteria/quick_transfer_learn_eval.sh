# For a given localization task, this script evaluates 18 finetuned models (each one using a different pretrained model).
# Only models from one checkpoint is evaluated (which differs from ./eval_one_task_all_epochs.sh)
# Other users are encourages to update: PARAM_FOLDER, VAL_LOGS, TASK_FOLDER.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  MODE=$1 # Breaks the training into partitions that can be run on separate terminals; options are [0, 1]
  TASK_SCAN=$2 # Modality the pretrained model is trained on (view quick_runs/README.md on TASK_SCAN for more)
  TASK_NUM=$3 # Localization task to evaluate (view quick_runs/README.md for more)
  EPOCH=$4 # The checkpoint to use for the finetuned model
  GPU=${5:-2}  # The GPU device to use (view quick_runs/README.md for more)
  ROOT=${6:-"/raid/home/slai16/research_retrain"} # Root folder containing parameters (and where to store evaluation logs)

  ITER=$(( 75000 + $EPOCH * 25000 ))
  PARAM_FOLDER="${ROOT}/transfer_learn_task_${TASK_SCAN}_${TASK_NUM}"
  VAL_LOGS="${ROOT}/eval_tl_finetune_${EPOCH}_epochs/eval_tl_task_${TASK_SCAN}_${TASK_NUM}"
  TASK_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"


  if [[ $MODE == 0 ]]; then
    echo "EVALUATING MODEL (TASK: SCAN $TASK_SCAN, LANDMARK $TASK_NUM) TRANSFER LEARNED FROM SET 1 FAT SCANS"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "EVALUATING MODEL TRANSFER LEARNED FROM FAT LANDMARK $counter"

      FILES="$TASK_FOLDER/val_image_paths.txt $TASK_FOLDER/val_landmark_paths.txt"
      PARAM_FILE="$PARAM_FOLDER/transfer_from_set_1_fat_$counter/dev/model-${ITER}"
      LOG_FOLDER="$VAL_LOGS/transfer_from_set_1_fat_$counter"

  #    echo "$PARAM_FILE"
  #    echo "$FILES"
  #    echo "$LOG_FOLDER"

      python DQN.py \
        --task eval  \
        --gpu ${GPU} \
        --load ${PARAM_FILE} \
        --files ${FILES} \
        --logDir ${LOG_FOLDER} \
        --saveGif \
        --agents 1
    done
  elif [[ $MODE == 1 ]]; then
    echo "EVALUATING MODEL (TASK: SCAN $TASK_SCAN, LANDMARK $TASK_NUM) TRANSFER LEARNED FROM SET 1 WATER SCANS"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "EVALUATING MODEL TRANSFER LEARNED FROM WATER LANDMARK $counter"

      FILES="$TASK_FOLDER/val_image_paths.txt $TASK_FOLDER/val_landmark_paths.txt"
      PARAM_FILE="$PARAM_FOLDER/transfer_from_set_1_water_$counter/dev/model-${ITER}"
      LOG_FOLDER="$VAL_LOGS/transfer_from_set_1_water_$counter"

  #    echo "$PARAM_FILE"
  #    echo "$FILES"
  #    echo "$LOG_FOLDER"

      python DQN.py \
        --task eval  \
        --gpu ${GPU} \
        --load ${PARAM_FILE} \
        --files ${FILES} \
        --logDir ${LOG_FOLDER} \
        --saveGif \
        --agents 1
    done
  fi

  exit
}