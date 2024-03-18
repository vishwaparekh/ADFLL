# This script trains DQN agent(s) using the dataset processed by quick_rungs/demo_rungs/parse_split_raw_data.sh.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  ROOT="/raid/home/slai16/Downloads/demo2"
  FILES="${ROOT}/train_image_paths.txt ${ROOT}/train_labels_path.txt"
  LOG_DIR="${ROOT}/train_logs"

  python DQN.py \
    --gpu 0 \
    --task train  \
    --files ${FILES} \
    --logDir ${LOG_DIR} \
    --lr 0.001 \
    --max_epochs 3 \
    --agents 9 \
    --no_erb

  exit
}