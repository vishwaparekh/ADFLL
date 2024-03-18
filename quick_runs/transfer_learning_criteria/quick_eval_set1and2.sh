# Evaluate the models trained from random initialization (to ensure they have converged)
# Other users are encourages to update: ROOT, PARAM_FOLDER, VAL_LOGS
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  MODE=$1 # Breaks the evaluation into partitions that can be run on separate terminals; options are [0, 1, 2, 3]
  EPOCH=${2:-3} # Checkpoint to use for evaluation; options are integers >= 1, but limited to available checkpoints

  ITER=$(( $EPOCH * 25000 ))
  # Folder containing dataset
  ROOT="/home/slai16/research/rl_registration_exp2/parsed_data"
  # Folder containing parameters to evaluated
  PARAM_FOLDER="/raid/home/slai16/research_retrain/train_log_36_models"
  # Folder to store evaluation results
  VAL_LOGS="/raid/home/slai16/research_retrain/eval_rand_init_models"


  if [[ $MODE == 0 ]]; then
    echo "Evaluating Set 1 Fat Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_1_fat_${counter}"

      FOLDER="${ROOT}/set_1_fat_landmark_${counter}"
      FILES="${FOLDER}/val_image_paths.txt ${FOLDER}/val_landmark_paths.txt"

      PARAM_FILE="${PARAM_FOLDER}/set_1_fat_${counter}/dev/model-${ITER}"
      LOG_FOLDER="${VAL_LOGS}/set_1_fat_landmark_${counter}"

#      echo "${PARAM_FILE}"
#      echo "${FILES}"
#      echo "${LOG_FOLDER}"

      python DQN.py \
        --task eval  \
        --gpu 0 \
        --load ${PARAM_FILE} \
        --files ${FILES} \
        --logDir ${LOG_FOLDER} \
        --saveGif \
        --agents 1
    done
  elif [[ $MODE == 1 ]]; then
    echo "Evaluating Set 1 Water Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_1_wtr_${counter}"

      FOLDER="${ROOT}/set_1_water_landmark_${counter}"
      FILES="${FOLDER}/val_image_paths.txt ${FOLDER}/val_landmark_paths.txt"

      PARAM_FILE="${PARAM_FOLDER}/set_1_wtr_${counter}/dev/model-${ITER}"
      LOG_FOLDER="${VAL_LOGS}/set_1_water_landmark_${counter}"

#      echo "${PARAM_FILE}"
#      echo "${FILES}"
#      echo "${LOG_FOLDER}"

      python DQN.py \
        --task eval  \
        --gpu 1 \
        --load ${PARAM_FILE} \
        --files ${FILES} \
        --logDir ${LOG_FOLDER} \
        --saveGif \
        --agents 1
    done
  elif [[ $MODE == 2 ]]; then
    echo "Evaluating Set 2 Fat Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_2_fat_${counter}"

      FOLDER="${ROOT}/set_2_fat_landmark_${counter}"
      FILES="${FOLDER}/val_image_paths.txt $FOLDER/val_landmark_paths.txt"

      PARAM_FILE="${PARAM_FOLDER}/set_2_fat_${counter}/dev/model-${ITER}"
      LOG_FOLDER="${VAL_LOGS}/set_2_fat_landmark_${counter}"

#      echo "${PARAM_FILE}"
#      echo "${FILES}"
#      echo "${LOG_FOLDER}"

      python DQN.py \
        --task eval  \
        --gpu 2 \
        --load ${PARAM_FILE} \
        --files ${FILES} \
        --logDir ${LOG_FOLDER} \
        --saveGif \
        --agents 1
    done
  elif [[ $MODE == 3 ]]; then
    echo "Evaluating Set 2 Water Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_2_wtr_${counter}"

      FOLDER="${ROOT}/set_2_water_landmark_${counter}"
      FILES="${FOLDER}/val_image_paths.txt ${FOLDER}/val_landmark_paths.txt"

      PARAM_FILE="${PARAM_FOLDER}/set_2_wtr_${counter}/dev/model-${ITER}"
      LOG_FOLDER="${VAL_LOGS}/set_2_water_landmark_${counter}"

#      echo "${PARAM_FILE}"
#      echo "${FILES}"
#      echo "${LOG_FOLDER}"

      python DQN.py \
        --task eval  \
        --gpu 3 \
        --load ${PARAM_FILE} \
        --files ${FILES} \
        --logDir ${LOG_FOLDER} \
        --saveGif \
        --agents 1
    done
  fi

  exit
}