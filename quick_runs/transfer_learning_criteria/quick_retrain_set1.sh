# Continues training 18 models with random initializations on training set 1; these models
# continue training from where they left off from the quick_train_fine_data.sh script.
# These models are used for transfer learning.
# Other users are encourages to update: ROOT, TRAIN_LOGS, PARAM_FOLDER.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  MODE=$1 # Breaks the training into partitions that can be run on separate terminals; options are [0, 1, 2, 3]
  ROOT=/home/slai16/research/rl_registration_exp2/parsed_data
  TRAIN_LOGS=/raid/home/slai16/research_retrain/retrain_set_1
  PARAM_FOLDER="/raid/home/slai16/research_retrain/train_log_36_models"

  if [[ $MODE == 0 ]]; then
    echo "Training Set 1 Fat Scans (Part 1)"
    for counter in 0 1 2 3; do
      echo "Starting set_1_fat_$counter"
      PARAM_FILE="$PARAM_FOLDER/set_1_fat_$counter/dev/model-75000"
      FOLDER="$ROOT/set_1_fat_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"

#      echo "${PARAM_FILE}"
#      echo "${TRAIN_LOGS}/set_1_fat_${counter}"
#      echo "${FILES}"

      python DQN.py \
        --task train  \
        --load $PARAM_FILE \
        --gpu 0 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_1_fat_$counter
    done
  elif [[ $MODE == 1 ]]; then
    echo "Training Set 1 Fat Scans (Part 2)"
    for counter in 4 5 6 7 8; do
      echo "Starting set_1_fat_$counter"
      PARAM_FILE="$PARAM_FOLDER/set_1_fat_$counter/dev/model-75000"
      FOLDER="$ROOT/set_1_fat_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"

#      echo "${PARAM_FILE}"
#      echo "${TRAIN_LOGS}/set_1_fat_${counter}"
#      echo "${FILES}"

      python DQN.py \
        --task train  \
        --load $PARAM_FILE \
        --gpu 1 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_1_fat_$counter
    done
  elif [[ $MODE == 2 ]]; then
    echo "Training Set 1 Water Scans (Part 1)"
    for counter in 0 1 2 3; do
      echo "Starting set_1_wtr_$counter"
      PARAM_FILE="$PARAM_FOLDER/set_1_wtr_$counter/dev/model-75000"
      FOLDER="$ROOT/set_1_water_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"

#      echo "${PARAM_FILE}"
#      echo "${TRAIN_LOGS}/set_1_wtr_${counter}"
#      echo "${FILES}"

      python DQN.py \
        --task train  \
        --load $PARAM_FILE \
        --gpu 2 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_1_wtr_$counter
    done
  elif [[ $MODE == 3 ]]; then
    echo "Training Set 1 Water Scans (Part 2)"
    for counter in 4 5 6 7 8; do
      echo "Starting set_1_wtr_$counter"
      PARAM_FILE="$PARAM_FOLDER/set_1_wtr_$counter/dev/model-75000"
      FOLDER="$ROOT/set_1_water_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"

#      echo "${PARAM_FILE}"
#      echo "${TRAIN_LOGS}/set_1_wtr_${counter}"
#      echo "${FILES}"

      python DQN.py \
        --task train  \
        --load $PARAM_FILE \
        --gpu 3 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_1_wtr_$counter
    done
  fi

  exit
}