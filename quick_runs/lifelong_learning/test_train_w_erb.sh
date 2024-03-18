# Training a DQN agent which is given an ERB to ensure loading buffer works after code merge.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  ROOT="/home/slai16/research/rl_registration_exp2/parsed_data"
  TRAIN_LOGS="/raid/home/slai16/research_retrain/test_load_erb"
  ERB_PATH="/raid/home/slai16/research_retrain/test_save_erb/dev/Experience_Replay_Buffer.obj"

  DATA_FOLDER="${ROOT}/set_1_fat_landmark_1"
  FILES="${DATA_FOLDER}/train_image_paths.txt ${DATA_FOLDER}/train_landmark_paths.txt"

  python DQN.py \
    --task train  \
    --gpu 0 \
    --files ${FILES} \
    --agents 1 \
    --logDir ${TRAIN_LOGS} \
    --prev_exp ${ERB_PATH}

  exit
}