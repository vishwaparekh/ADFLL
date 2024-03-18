# This script demos preprocess_data.py, which splits the dataset into train and validation then parses the image
# and landmark files into the format expected by DQN.py.
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  IMGS_FOLDER="/home/slai16/research/rl_registration/MD_RL_nii_F_10052021"
  IMG_FILENAME="image_paths"
  LABELS_PATH="/home/slai16/research/rl_registration/RL_landmarks_10052021.xlsx"
  LABEL_FILENAME="labels_path"
  OUTPUT_FOLDER="/raid/home/slai16/Downloads/demo2"
  TRAIN_SPLIT=20
  TEMPLATE="F_MRI_*"

  python registration_utils/preprocess_data.py \
    --imgs ${IMGS_FOLDER} \
    --img_file ${IMG_FILENAME} \
    --labels ${LABELS_PATH} \
    --label_file ${LABEL_FILENAME} \
    --output ${OUTPUT_FOLDER} \
    --train_split ${TRAIN_SPLIT} \
    --custom_names ${TEMPLATE}

  exit
}