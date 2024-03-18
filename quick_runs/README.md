## Overview

These scripts are entry points to running segments of experiments. Many of them accept command line
arguments. The common command line arguments will be documented here in detail:

* `TASK_SCAN` Select the modality of the scan (water or fat)
* `TASK_NUM` Select what localization task is being performed. There are 9 available anatomical landmark labels provided for each patient, so the value inputs are integers from [0, 8]  
* `GPU` Which GPU device to run the script in; for a machine with 4 GPUs, value inputs are integers from [0, 3]

# Task Numbers

For simplicity, the 9 localization tasks have been mapped to numbers. The following is 
the mapping: 

0) Left Knee 
1) Right Knee
2) Left Trocahnter
3) Right Trocahnter
4) Left Kidney
5) Right Kidney
6) Spleen
7) Left Lung
8) Right Lung