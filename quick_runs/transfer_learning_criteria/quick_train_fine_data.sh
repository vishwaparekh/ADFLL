# Trains 36 models with random initialization (to be used for transfer learning and baselines).
# Models to be used for transfer learning are trained using data from training set 1.
# Models to be used for baseline comparisons are trained using data from training set 2.
# Other users are encourages to update: ROOT, TRAIN_LOGS
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  MODE=$1 # Breaks the training into partitions that can be run on separate terminals; options are [0, 1, 2, 3]
  ROOT=/home/slai16/research/rl_registration_exp2/parsed_data
  TRAIN_LOGS=/raid/home/slai16/research_retrain/train_36_model

  if [[ $MODE == 0 ]]; then
    echo "Training Set 1 Fat Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_1_fat_$counter"
      FOLDER="$ROOT/set_1_fat_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"
      python DQN.py \
        --task train  \
        --gpu 0 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_1_fat_$counter
    done
  elif [[ $MODE == 1 ]]; then
    echo "Training Set 1 Water Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_1_wtr_$counter"
      FOLDER="$ROOT/set_1_water_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"
      python DQN.py \
        --task train  \
        --gpu 1 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_1_wtr_$counter
    done
  elif [[ $MODE == 2 ]]; then
    echo "Training Set 2 Fat Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_2_fat_$counter"
      FOLDER="$ROOT/set_2_fat_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"
      python DQN.py \
        --task train  \
        --gpu 2 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_2_fat_$counter
    done
  elif [[ $MODE == 3 ]]; then
    echo "Training Set 2 Water Scans"
    for counter in 0 1 2 3 4 5 6 7 8; do
      echo "Starting set_2_wtr_$counter"
      FOLDER="$ROOT/set_2_water_landmark_$counter"
      FILES="$FOLDER/train_image_paths.txt $FOLDER/train_landmark_paths.txt"
      python DQN.py \
        --task train  \
        --gpu 3 \
        --files $FILES \
        --agents 1 \
        --logDir $TRAIN_LOGS/set_2_wtr_$counter
    done
  fi

  exit
}