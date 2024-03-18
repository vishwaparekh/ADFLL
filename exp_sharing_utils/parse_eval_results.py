'''
Process and aggregate the prediction files outputted by our DQN experiments (mainly experience sharing)
so that it can be easily exported to an excel or csv file, allowing us to further analyze the results.

Author: Shuhao Lai <shuhaolai18@gmail.com>
'''

import re
import os
import numpy as np


def extract_dist(str_res):
    '''
    Parse a human-readable string containing predictions results
    
    :param str_res: The string to parse out predictions results
    :return: Float containing xy distance, z dist, and euclidean distance between ground truth and prediction.
    '''
    dist_pattern = 'XY Dist (\d+.\d+) mm, Z Dist (\d+.\d+) mm, Dist (\d+.\d+) mm'
    match_obj = re.search(dist_pattern, str_res)
    xy_dist = float(match_obj.group(1))
    z_dist = float(match_obj.group(2))
    dist = float(match_obj.group(3))
    return xy_dist, z_dist, dist


def parse_eval_results(file_path):
    '''
    For each line of results from the provided text file, parse out Euclidean distance of prediction to ground truth
    and save as a numpy vector. 
    
    :param file_path: Text file containing prediction results. 
    :return: Numpy array containing euclidean distance of predictions to ground truth. 
    '''
    
    with open(file_path) as file:
        # 10 patients in test set
        exp_res = np.zeros(10)
        idx = 0
        for file_name in file:
            str_res = next(file)
            xy_dist, z_dist, dist = extract_dist(str_res)
            exp_res[idx] = dist
            idx += 1
        return exp_res


def parse_eval_pseudo_label_models(base_folder):
    '''
    Get Euclidean distance for predictions from n prediction files. n is equal to the number of directories
    in the base folder; each directory contains a evaluation run of our DQN agent(s).

    This function is particularly useful to evaluated the many prediction runs needed to generate
    pseudo labels for many localization tasks. When generating pseudo labels, we typically use
    one agent/task per evaluation run, so each prediction file contains localization predictions for
    a single task on ~10 patients.
    
    :param base_folder: Folder containing n evaluation runs of our DQN agent. 
    :return: 2D array of Euclidean distance results with shape (num. patients, num eval runs) and a
        list of folder names corresponding to each column.
    '''
    
    task_folders = list(os.listdir(base_folder))
    eval_folder_name = 'eval_pseudo_labels_model'
    eval_folders = [os.path.join(base_folder, task_folder, eval_folder_name) for task_folder in task_folders]
    all_eval_arr, eval_foldernames = generic_parse_eval_results(eval_folders)
    return all_eval_arr, eval_foldernames


def generic_parse_parent_eval_folder(parent_eval_folder):
    '''
    Helper function to generate a list of folders to parse using the function generic_parse_eval_results(.).

    :param parent_eval_folder: The parent folder to find other folders we want to parse.
    :return: 2D array of Euclidean distance results with shape (num. patients, num eval runs) and a
        list of folder names corresponding to each column.
    '''

    eval_folders = list(os.listdir(parent_eval_folder))
    eval_folders = [os.path.join(parent_eval_folder, eval_folder) for eval_folder in eval_folders]
    all_eval_arr, eval_foldernames = generic_parse_eval_results(eval_folders)
    return all_eval_arr, eval_foldernames


def generic_parse_eval_results(eval_folders):
    '''
    For each folder in eval_folders, we access the prediction file terminal_landmarks.txt and parse
    out the landmarks' Euclidean distance to the ground truth. The results from each file
    are stored in a numpy 2D array.

    :param eval_folders: The parent folder to find other folders, each containing a prediction file.
    :return: A 2D numpy array of Euclidean distances and a list of folder
        names that correspond to each column of the 2D array.
    '''

    eval_folders.sort()
    all_eval_res = [] # Each row represents results from one eval run
    for i, eval_folder in enumerate(eval_folders):
        eval_path = os.path.join(eval_folder, 'terminal_landmarks.txt')
        exp_res = parse_eval_results(eval_path)
        all_eval_res.append(exp_res)

    all_eval_arr = np.array(all_eval_res) # Has shape (num eval runs, num patients)
    all_eval_arr = all_eval_arr.T # Has shape (num patients, num eval runs)

    eval_folder_names = [os.path.basename(folder) for folder in eval_folders]
    return all_eval_arr, eval_folder_names


def parse_tl_results(folder):
    '''
    This function is similar to generic_parse_eval_results(.) but parses out the results from the transfer learning
    experiments where 10 patients are used and 18 evaluation runs were done (one for each pair of landmark and
    scan type -->  9 * 2 = 18).

    :param folder: Parent folder containing a folder for each transfer learning evaluation run.
    :return: 2D matrix containing Euclidean distances for 10 rows/patients and 18 cols/evaluation runs.
    '''

    tl_folders = list(os.listdir(folder))
    # Since landmarks only extend from 0-8 and fat appears alphabetically before water,
    # default sort works
    tl_folders.sort()
    # 18 experiments per folder, each experiment has 10 patients in test set
    all_exps_res = np.zeros((10, 18))
    for i, tl_folder in enumerate(tl_folders):
        eval_path = os.path.join(folder, tl_folder, 'terminal_landmarks.txt')
        exp_res = parse_eval_results(eval_path)
        all_exps_res[:, i] = exp_res
    return all_exps_res


def parse_baseline_results(file_path):
    '''
    Parses the text file containing results from a baseline model (for the transfer learning experiment);
    the baseline is simply a model trained WITHOUT pretrained parameters (trained from random initialization).

    :param file_path: Path to baseline results.
    :return: Numpy array of shape (num. patients, 1) containing baseline results.
    '''

    exp_res = parse_eval_results(file_path)
    exp_res = exp_res.reshape((-1, 1))
    return exp_res


def print_formated_eval_res(tl_folder, baseline_file):
    '''
    Helper function to parse out the transfer learning results for a given task (we finetune on the given task
    using pretrained parameters from different tasks and modalities). The evaluation results
    from the baseline is also parsed. The results are stored in a 2D matrix with shape
    (num. patients, num task and modality pairs + baseline). The 2D array is then printed out
    so that it can simply be copied and pasted into a csv file or excel file to be further processed.

    :param tl_folder: The folder containing transfer learning evaluation runs.
    :param baseline_file: The file containing the baseline evaluation run.
    '''

    tl_res = parse_tl_results(tl_folder)
    baseline_res = parse_baseline_results(baseline_file)
    all_res = np.hstack([baseline_res, tl_res])
    for row in all_res:
        for i, col in enumerate(row):
            if i == 18:
                print(col, end='')
            else:
                print(col, end=',')
        print()


def print_eval_tl_res():
    '''
    WARNING: depreciated

    This function serves a similar purpose to summarize_tl_eval_results() and is depreciated. It will be removed
    once other functions using it have been updated.
    '''

    epoch = 3
    task_nums = [6, 4, 3, 0, 8]
    lrs = ['0.0002', '0.0004', '0.0006', '0.0008']

    for lr in lrs:
        for task_num in task_nums:
            print('--------------------------------------------------------')
            print(f'Task Num = {task_num}, LR = {lr}, Epoch = {epoch}')
            tl_folder = f'/raid/home/slai16/research_retrain/Exp_TL_LR_{lr}/eval_tl_finetune_{epoch}_epochs/eval_tl_task_fat_{task_num}'
            baseline_file = f'/raid/home/slai16/research_retrain/eval_baseline_epochs_1_2_3/eval_baseline_train_{epoch}_epochs/task_fat_{task_num}/terminal_landmarks.txt'
            print_formated_eval_res(tl_folder, baseline_file)


def print_matrix(mat):
    '''
    Generic function to print a matrix so that it can be transferred to a csv or excel file.

    :param mat: Matrix to print.
    '''

    nrows, ncols = len(mat), len(mat[0])
    for row in mat:
        for i, col in enumerate(row):
            if i == ncols - 1:
                print(col, end='')
            else:
                print(col, end=',')
        print()


def summarize_tl_eval_results():
    '''
    This function aggregates the results of pretraining on 5 tasks (spleen, left kidney, right torchanter, left knee, and right lung)
    and 3 different checkpoints during training. For a given task and checkpoint, this function creates a
    2D matrix with shape (num. pretrained tasks, num. learning rates), so a given pretrained task and learning rate
    coordinate will output the average performance. There are 18 available models to pretrained from (9 landmarks vs 2 modalities)
    and the available learning rates are [0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001].

    Note, there are 5 tasks and 3 checkpoints, resulting in 15 2D matrices, each with 18 x 5 elements. Each
    matrix are printed in a format such that they can be easily exported into a csv or excel file for further processing.
    '''

    EPOCH = 3
    TASKS = {
        6: 'spleen',
        4: 'left kidney',
        3: 'right trochantor',
        0: 'left knee',
        8: 'right lung'
    }
    LRS = ['0.0001', '0.0002', '0.0004', '0.0006', '0.0008', '0.001']
    EXPS = [3, 4, 5, 6, 7, 2]  # Experiment numbers, useful for folder names
    NUM_PRETRAIN_TASK = 9
    NUM_LR = len(LRS)

    for task_num, task_name in TASKS.items():
        task_across_lr_fat = np.zeros((NUM_PRETRAIN_TASK, NUM_LR))
        task_across_lr_water = np.zeros((NUM_PRETRAIN_TASK, NUM_LR))

        for i, (lr, exp) in enumerate(zip(LRS, EXPS)):
            tl_folder = f'/raid/home/slai16/research_retrain/Exp_{exp}_TL_LR_{lr}/eval_tl_finetune_{EPOCH}_epochs/eval_tl_task_fat_{task_num}'
            tl_res = parse_tl_results(tl_folder)  # Matrix has shape Num. Patients x 18 TL Models

            fat_tl_res = tl_res[:, :9]
            water_tl_res = tl_res[:, 9:]

            avg_fat_tl_res = np.mean(fat_tl_res, axis=0)
            avg_water_tl_res = np.mean(water_tl_res, axis=0)

            task_across_lr_fat[:, i] = avg_fat_tl_res
            task_across_lr_water[:, i] = avg_water_tl_res

        baseline_file = f'/raid/home/slai16/research_retrain/eval_baseline_epochs_1_2_3/eval_baseline_train_{EPOCH}_epochs/task_fat_{task_num}/terminal_landmarks.txt'
        baseline_res = parse_baseline_results(baseline_file)  # Matrix has shape Num. Patients x 1
        avg_baseline = np.mean(baseline_res)

        print('--------------------------------------------------------')
        print(f'Task Name = {task_name} (Task Num {task_num}), Epoch = {EPOCH}, FAT pretrained models')
        print_matrix(task_across_lr_fat)

        print('--------------------------------------------------------')
        print(f'Task Name = {task_name} (Task Num {task_num}), Epoch = {EPOCH}, WATER pretrained models')
        print_matrix(task_across_lr_water)

        print('--------------------------------------------------------')
        print(f'Task Name = {task_name} (Task Num {task_num}), Epoch = {EPOCH}, BASELINE')
        print(avg_baseline)


def summarize_pseudo_eval_results():
    '''
    Aggregate results from using pseudo labels to train a model. The results include:

        (1) Euclidean distance to ground truth and predictions from model trained with random initializations and ground truth labels
        (2) Euclidean distance to ground truth and predictions from model trained with random initializations and pseudo labels
        (3) Euclidean distance between pseudo labels and their ideal pseudo labels (hand annotated)

    The aggregated results are stored in a matrix and printed to be exported in a csv or excel file.
    '''

    gt_labels_eval_folder = '/raid/home/slai16/research_retrain/eval_rand_init_models'
    pseudo_labels_folder = '/raid/home/slai16/research_retrain/pseudo_labels'

    eval_gt_labels, eval_folders_gt_labels = generic_parse_parent_eval_folder(gt_labels_eval_folder)
    pseudo_labels_quality, pseudo_labels_folders = generic_parse_parent_eval_folder(pseudo_labels_folder)
    eval_pseudo_labels, eval_folders_pseudo_labels = parse_eval_pseudo_label_models(pseudo_labels_folder)

    print('Printing results for models with random initializations trained on GT labels')
    print('Columns:')
    print(eval_folders_gt_labels)
    print('Eval Results:')
    print_matrix(eval_gt_labels)

    print('------'*6)

    print('Printing results for models with random initializations trained on psuedo labels')
    print('Columns:')
    print(eval_folders_pseudo_labels)
    print('Eval Results:')
    print_matrix(eval_pseudo_labels)

    print('------'*6)

    print('Printing pseudo label quality')
    print('Columns:')
    print(pseudo_labels_folders)
    print('Eval Results:')
    print_matrix(pseudo_labels_quality)


if __name__=='__main__':
    # summarize_tl_eval_results()
    # summarize_pseudo_eval_results()
    pass