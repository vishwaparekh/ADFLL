# This script runs the eval_pseudo_label_models.sh script on all 9 localization tasks (only on fat scans).

{
  eval "$(conda shell.bash hook)"
  conda activate rl_medical_env_2021

  for task in 0 1 2 3 6 7 8; do
    bash quick_runs/exp_sharing_pseudo_labels/eval_pseudo_label_models.sh fat ${task}
  done

  exit
}