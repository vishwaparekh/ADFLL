# Training a DQN agent with random initalizations to ensure ERB is saving properly after code merge.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  ROOT="/home/slai16/research/rl_registration_exp2/parsed_data"
  TRAIN_LOGS="/raid/home/slai16/research_retrain/test_save_erb"

  DATA_FOLDER="${ROOT}/set_1_fat_landmark_0"
  FILES="${DATA_FOLDER}/train_image_paths.txt ${DATA_FOLDER}/train_landmark_paths.txt"

  python DQN.py \
    --task train  \
    --gpu 0 \
    --files ${FILES} \
    --agents 1 \
    --logDir ${TRAIN_LOGS}

  exit
}