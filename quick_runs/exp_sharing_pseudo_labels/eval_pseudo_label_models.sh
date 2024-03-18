# This script evaluates the models trained with pseudo labels.
# Other users are encourages to update: PARAM_FOLDER, DATA_FOLDER, LOGS_ROOTS.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  # Command line arguments
  TASK_SCAN=$1 # Type of scan to use for evaluation (options are 'fat' and 'water') (view quick_runs/README.md for more)
  TASK_NUM=$2 # Localization task to use for evaluation (view quick_runs/README.md for more)
  EPOCH=${3:-3} # What epoch to use parameters from (options are [1, 2, 3])
  GPU=${4:-0} # Which GPU to use (view quick_runs/README.md for more)
  LOGS_ROOT=${5:-/raid/home/slai16/research_retrain/pseudo_labels} # Folder to save evaluation logs (prediction text file, images, and gifs)

  ITER=$(( $EPOCH * 25000 ))
  # PARAM_FOLDER is the path to the training log folder for a model trained using pseudo labels
  PARAM_FOLDER="/raid/home/slai16/research_retrain/pseudo_labels/set2_task_${TASK_SCAN}_${TASK_NUM}/pseudo_label_training_logs"
  # DATA_FOLDER is the path to the folder storing the dataset
  DATA_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/set_2_${TASK_SCAN}_landmark_${TASK_NUM}"
  # LOG_FOLDER is the folder path to store evaluation results
  LOG_FOLDER="${LOGS_ROOT}/set2_task_${TASK_SCAN}_${TASK_NUM}/eval_pseudo_labels_model"

  DATA_FILES="${DATA_FOLDER}/val_image_paths.txt ${DATA_FOLDER}/val_landmark_paths.txt"
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