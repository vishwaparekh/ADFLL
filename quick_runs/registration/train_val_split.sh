eval "$(conda shell.bash hook)"
conda activate rl_medical_env_2021
python ./registration_utils/preprocess_data.py \
  --imgs /home/slai16/research/rl_registration/MD_RL_nii_F_10052021/ /home/slai16/research/rl_registration/MD_RL_nii_W_11042021/ \
  --labels /home/slai16/research/rl_registration/RL_landmarks_10052021.xlsx /home/slai16/research/rl_registration/RL_landmarks_10052021.xlsx \
  --output /home/slai16/research/rl_registration_exp2/ \
  --custom_names F_MRI_* W_MRI_*
