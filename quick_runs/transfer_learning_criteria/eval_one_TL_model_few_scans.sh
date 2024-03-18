# This script evaluates one TL model (given a goal task, a pretrained task, a checkpoint, and a learning rate)
# on one scan (specified by text file from FILES variable).
# Other users are encourages to update: ROOT, VAL_LOGS, TASK_FOLDER
#
# Author: Shuhao Lai <shuhaolai18@gmail.com>

{
  # This script is useful to time evaluation runs for a single image.
  # In particular, we timed evaluation for fat left knee task TL from fat left knee
  # finetuned for 1 epoch with LR 0.0001 for patient MD001 (from test set).
  # User must manually adjust task text files (image paths and landmark paths) to select which patients to evaluate
  # To adjust learning rate, change the ROOT variable

  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  TASK_SCAN=fat # Modality of task to evaluate (view quick_runs/README.md for more)
  TASK_NUM=0 # Localization task to evaluate (view quick_runs/README.md for more)
  TL_SCAN=fat # Modality the pretrained model is trained on
  TL_NUM=0 # Task the pretrained model is trained on
  EPOCH=1 # The checkpoint to use for the finetuned model
  GPU=0 # The GPU device to use (view quick_runs/README.md for more)
  ROOT="/raid/home/slai16/research_retrain/Exp_3_TL_LR_0.0001" # The folder containing the parameters for the finetuned model (note that this
    # is also specifies the learning rate used to finetune the model)

  ITER=$(( 75000 + $EPOCH * 25000 ))
  PARAM_FOLDER="${ROOT}/transfer_learn_task_${TASK_SCAN}_${TASK_NUM}"
  VAL_LOGS="${ROOT}/timing/eval_tl_finetune_${EPOCH}_epochs/eval_tl_task_${TASK_SCAN}_${TASK_NUM}"
  TASK_FOLDER="/home/slai16/research/rl_registration_exp2/parsed_data/test_set_time_exp"

  FILES="$TASK_FOLDER/val_image_paths.txt $TASK_FOLDER/val_landmark_paths.txt"
  PARAM_FILE="$PARAM_FOLDER/transfer_from_set_1_${TL_SCAN}_${TL_NUM}/dev/model-${ITER}"
  LOG_FOLDER="$VAL_LOGS/transfer_from_set_1_${TL_SCAN}_${TL_NUM}"

#  echo "GPU: ${GPU}"
#  echo "Param File: ${PARAM_FILE}"
#  echo "Task Files: ${FILES}"
#  echo "Log Dir: ${LOG_FOLDER}"

  python DQN.py \
    --task eval  \
    --gpu ${GPU} \
    --load ${PARAM_FILE} \
    --files ${FILES} \
    --logDir ${LOG_FOLDER} \
    --saveGif \
    --agents 1
}