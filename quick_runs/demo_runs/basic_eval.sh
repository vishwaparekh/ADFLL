# This script demos the evaluation step from DQN agent(s) trained by quick_runs/demo_runs/basic_train.sh.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  ROOT="/raid/home/slai16/Downloads/demo"
  PARAM_FOLDER="${ROOT}/prepared_train_logs"

  FILES="${ROOT}/val_image_paths.txt ${ROOT}/val_labels_path.txt"
  PARAM_FILE="${PARAM_FOLDER}/dev/model-75000"
  VAL_LOGS="${ROOT}/eval_log"

  python DQN.py \
    --task eval  \
    --gpu 1 \
    --load ${PARAM_FILE} \
    --files ${FILES} \
    --logDir ${VAL_LOGS} \
    --saveGif \
    --agents 9
}