# Evaluates a baseline model (model trained on training set 2 but with random initializations).
# Other users are encourages to update: TASK_FOLDER, FILES, LOG_FOLDER, PARAM_FILE
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  TASK_SCAN=$1 # Modality of task to evaluate (view quick_runs/README.md for more)
  TASK_NUM=$2 # Localization task to evaluate (view quick_runs/README.md for more)
  EPOCH=$3 # The checkpoint to evaluate
  GPU="${4:-3}" # The GPU device to use (view quick_runs/README.md for more)

  ITER=$(( $EPOCH * 25000 ))
  TASK_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"
  FILES="$TASK_FOLDER/val_image_paths.txt $TASK_FOLDER/val_landmark_paths.txt"
  LOG_FOLDER="/raid/home/slai16/research_retrain/eval_baseline_train_${EPOCH}_epochs/task_${TASK_SCAN}_${TASK_NUM}"


  if [[ $TASK_SCAN == "water" ]]; then
    # Required due to inconsistent use of water and wtr
    PARAM_FILE="/raid/home/slai16/research_retrain/train_log_36_models/set_2_wtr_${TASK_NUM}/dev/model-${ITER}"
  else
    PARAM_FILE="/raid/home/slai16/research_retrain/train_log_36_models/set_2_${TASK_SCAN}_${TASK_NUM}/dev/model-${ITER}"
  fi


#  echo "$FILES"
#  echo "$PARAM_FILE"
#  echo "$LOG_FOLDER"
#  echo "$GPU"

  python DQN.py \
    --task eval  \
    --gpu ${GPU} \
    --load $PARAM_FILE \
    --files $FILES \
    --logDir $LOG_FOLDER \
    --saveGif \
    --agents 1

  exit
}