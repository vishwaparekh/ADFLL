'''
Split the 30 patients into 20-10 splits; 20 patients for the training set and 10 patients for the validation set.
The data are parsed into the format as expected by our DQN.

Author: Shuhao Lai <shuhaolai18@gmail.com>
'''

import pandas as pd
import numpy as np
import warnings
import argparse
import os
import re


class CustomFilenames:
    def __init__(self, template):
        self.template = template

    def __call__(self, id):
        return self.template.replace('*', id)


def get_filename_generators(templates, num_needed):
    if templates:
        assert len(templates) == num_needed
        gen_filenames = [CustomFilenames(template) for template in args.custom_names]
    else:
        gen_filenames = []
        for i in range(num_needed):
            template = f'MRI_{i}_*'
            gen_filenames.append(CustomFilenames(template))
    return gen_filenames


def save_paths_to_file(paths, output_folder, filename, train_split=20, seed=0):
    '''
    Saves paths to text file after splitting data into train and test (stratified by patient)
    '''

    if 0 < train_split < 1:
        N = len(paths)
        train_split = int(np.round(N * train_split))

    num_patients = len(paths[0])
    num_versions = len(paths)
    indices = np.arange(num_patients)

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    train_indices.sort()
    val_indices.sort()

    train_file_path = os.path.join(output_folder, f'train_{filename}.txt')
    f = open(train_file_path, "w")
    for patient_i in train_indices:
        for version_i in range(num_versions):
            f.write(paths[version_i][patient_i] + '\n')
    f.close()

    val_file_path = os.path.join(output_folder, f'val_{filename}.txt')
    f = open(val_file_path, "w")
    for patient_i in val_indices:
        for version_i in range(num_versions):
            f.write(paths[version_i][patient_i] + '\n')
    f.close()


def get_num(filename):
    num_pattern = '\d\d\d'
    num_str = re.search(num_pattern, filename).group()
    return int(num_str)


def get_all_image_paths(image_folders, output_folder, output_filename, train_split, gen_filename_funcs):
    '''
    Gets all image paths from many folders. Each folder contain scans for the same patients but with different techniques.
    Then saves all paths to a text file after splitting to train and val.
    '''
    file_paths = []  # Grouping paths from the same patient together
    for image_folder, gen_filename in zip(image_folders, gen_filename_funcs):
        file_paths.append(get_image_paths(image_folder, gen_filename))
    save_paths_to_file(file_paths, output_folder, output_filename, train_split=train_split)


def get_image_paths(image_folder, gen_filename):
    '''
    Returns list of image files from image folder after renaming and sorting them.

    Paths are sorted by patient number.
    '''

    image_filenames = os.listdir(image_folder)
    image_filenames = sorted(image_filenames, key=get_num)
    id_pattern = 'MD\d\d\d'
    paths = []
    for filename in image_filenames:
        if gen_filename:
            old_path = os.path.join(image_folder, filename)

            match_obj = re.search(id_pattern, filename)
            id = match_obj.group()
            filename = f'{gen_filename(id)}.nii'

            new_path = os.path.join(image_folder, filename)
            os.rename(old_path, new_path)
            paths.append(os.path.abspath(new_path))
        else:
            image_path = os.path.join(image_folder, filename)
            paths.append(os.path.abspath(image_path))
    return paths


def parse_all_annotations(annotation_paths, output_folder, output_filename, train_split, gen_filename_funcs):
    '''
    Parses many files of annotations then saves all paths to a text file after splitting to train and val.
    Each file contain annotations for the same patients but their scans are obtained using different techniques.
    '''

    file_paths = []  # Grouping paths from the same patient together
    for annotation_path, filename in zip(annotation_paths, gen_filename_funcs):
        file_paths.append(parse_annotations(annotation_path, output_folder, filename))
    save_paths_to_file(file_paths, output_folder, output_filename, train_split=train_split)


def parse_annotations(annotations_path, output_folder, gen_filename):
    '''
    Parses all rows of an annotation file
    Returns the list of paths where parsed text files are written.

    Paths are sorted by patient number
    '''

    try:
        df = pd.read_excel(annotations_path)
    except ValueError:
        df = pd.read_csv(annotations_path)

    file_paths = []
    for index, row in df.iterrows():
        file_path = parse_row(row, output_folder, gen_filename)
        abs_file_path = os.path.abspath(file_path)
        file_paths.append(abs_file_path)
    return sorted(file_paths, key=get_num)


def parse_row(row, output_folder, gen_filename, num_annotations=9):
    '''
    Parses a row of annotations by saving all annotation to a text file
    '''

    name = str(row[0])

    if not re.match('MD\d\d\d', name):
        warnings.warn("The following patient identifier is not well formatted: " + name)

    filename = gen_filename(name) if gen_filename else f'MRI_{name}'
    file_path = os.path.join(output_folder, f'{filename}.txt')
    f = open(file_path, "w")

    pattern = '(\d+),(\d+),(\d+)'
    all_digits = '\d+'
    for i in range(2, 2 + num_annotations):
        data = str(row[i])
        data = re.sub('\s', "", data)
        match_obj = re.match(pattern, data)
        if match_obj:
            data_line = f'{match_obj.group(1)},{match_obj.group(2)},{match_obj.group(3)}'
            f.write(data_line + '\n')
        elif re.match(all_digits, data) and (len(data) == 8 or len(data) == 9):
            # Excel converted text into number: 12,123,123 --> 12123123
            end_nums = 3 if len(data) == 9 else 2
            data_line = f'{data[0:end_nums]},{data[end_nums:end_nums+3]},{data[end_nums+3:end_nums+6]}'
            f.write(data_line + '\n')
        else:
            f.write(data + '\n')
            warnings.warn(f'Patient identifier {name} does not have well formatted data: {data}')

    f.close()
    return file_path


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parses the annotations and images into the correct format for MIDRL')
    parser.add_argument('-i', '--imgs', nargs='+', required=True, help="Path to the folder containing the images")
    parser.add_argument('-p', '--img_file', default='image_paths', help="Filename for file containing paths for images")
    parser.add_argument('-l', '--labels', nargs='+', required=True, help="Path to the excel file containing the annotations")
    parser.add_argument('-f', '--label_file', default='landmark_paths', help="Filename for file containing paths for annotations")
    parser.add_argument('-o', '--output', required=True, help="Path to the folder to store the parsed data")
    parser.add_argument('-t', '--train_split', type=int, default=20, help="Number of patients in training set")
    parser.add_argument('-c', '--custom_names', nargs='*', help='Custom name templates for the annotation and image files')
    args = parser.parse_args()

    assert len(args.imgs) == len(args.labels)
    if not os.path.isdir(args.output): os.mkdir(args.output)

    gen_filenames = get_filename_generators(args.custom_names, len(args.imgs))
    parse_all_annotations(args.labels, args.output, args.label_file, args.train_split, gen_filenames)
    get_all_image_paths(args.imgs, args.output, args.img_file, args.train_split, gen_filenames)