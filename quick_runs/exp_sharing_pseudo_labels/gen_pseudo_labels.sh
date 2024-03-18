# This script runs evaluation using DQN models (trained on training set 1) on training set 2 to generate pseudo labels.
# These pseudo labels can be used to trained another model, which does not have access to the original data.
# Other users are encourages to update: PARAM_FOLDER, DATA_FOLDER, LOGS_ROOTS.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  TASK_SCAN=${1:-fat} # Type of scan to generate pseudo labels for (options are 'fat' and 'water') (view quick_runs/README.md for more)
  TASK_NUM=$2 # Localization task to generate pseudo labels for (view quick_runs/README.md for more)
  EPOCH=${3:-3} # What epoch to use parameters from
  GPU=${4:-0} # Which GPU to use (view quick_runs/README.md for more)
  LOGS_ROOT=${5:-/raid/home/slai16/research_retrain/pseudo_labels} # Folder to save pseudo labels

  ITER=$(( $EPOCH * 25000 ))
  PARAM_FOLDER="/raid/home/slai16/research_retrain/train_log_36_models/set_1_${TASK_SCAN}_${TASK_NUM}"
  DATA_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"
  LOG_FOLDER="${LOGS_ROOT}/set2_task_${TASK_SCAN}_${TASK_NUM}"

  DATA_FILES="${DATA_FOLDER}/train_image_paths.txt ${DATA_FOLDER}/train_landmark_paths.txt"
  PARAM_FILE="$PARAM_FOLDER/dev/model-${ITER}"

#  echo "GPU: ${GPU}"
#  echo "Param File: ${PARAM_FILE}"
#  echo "Data Files: ${DATA_FILES}"
#  echo "Log Dir: ${LOG_FOLDER}"

  python DQN.py \
    --task eval  \
    --gpu ${GPU} \
    --load ${PARAM_FILE} \
    --files ${DATA_FILES} \
    --logDir ${LOG_FOLDER} \
    --saveGif \
    --agents 1

  exit
}