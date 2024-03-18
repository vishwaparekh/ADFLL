'''
Split the 30 patients into 10-10-10 splits for transfer learning experiment and saves the training data
in the format expected by our DQN model. Each patient will have data for 9 landmarks and 2 modalities (fat and water).

Author: Shuhao Lai <shuhaolai18@gmail.com>
'''

import pandas as pd
import numpy as np
import argparse
import os
import re


class CustomFilenames:
    '''
    Class that generates filenames using provided template and unique ID.
    '''
    def __init__(self, template):
        '''
        :param template: String template where "*" symbols are replaced.
        '''
        self.template = template

    def __call__(self, id):
        '''
        :param id: ID to use to generate filename.
        '''

        return self.template.replace('*', id)


def get_filename_generators(templates, num_needed):
    '''
    Create a list of CustomFilenames objects using default templates or provided templates.

    :param templates: List of templates to use (can be None).
    :param num_needed: Number of templates needed.
    :return: List of CustomFilenames objects.
    '''

    if templates:
        assert len(templates) == num_needed
        gen_filenames = [CustomFilenames(template) for template in args.custom_names]
    else:
        gen_filenames = []
        for i in range(num_needed):
            template = f'MRI_{i}_*'
            gen_filenames.append(CustomFilenames(template))
    return gen_filenames


class PatientData:
    '''
    Class representing a collection of patient data.
    '''

    def __init__(self, landmark_path, image_folder, filename_gen):
        '''
        :param landmark_path: Path to landmarks for patients.
        :param image_folder: Folder containing nii formatted images.
        :param filename_gen: CustomFilenames object to generate filenames.
        '''

        self.landmark_path = landmark_path
        self.image_folder = image_folder
        self.filename_gen = filename_gen

        self.id2landmarks = self.get_landmark_map(landmark_path)
        self.id2img_paths = self.get_img_map(image_folder)
        self.scan_type = self.get_scan_type(image_folder)

    def get_scan_type(self, image_folder):
        '''
        For a given image folder, parse the folder name to determine what modality the images
        are in (fat or water).

        :param image_folder: Path to the folder of images.
        :return: Modality of images in folder.
        '''

        dummy_file_path = os.path.join(image_folder, 'dummy.file')
        basename = os.path.basename(os.path.dirname(dummy_file_path))
        basename = basename.lower()
        if 'water' in basename or '_w_' in basename:
            return 'water'
        elif 'fat' in basename or '_f_' in basename:
            return 'fat'
        else:
            return None

    def get_img_map(self, image_folder):
        '''
        For images in a given image folder, generate a dictionary mapping patient ID to image path.

        :param image_folder: Folder containing all images to generate map for.
        :return: Dictionary mapping patient ID to image path.
        '''

        id2img_path = {}
        filenames = os.listdir(image_folder)
        id_pattern = 'md(\d\d\d)'
        for filename in filenames:
            match_obj = re.search(id_pattern, filename.lower())
            if match_obj:
                id = int(match_obj.group(1))
                img_path = os.path.join(image_folder, filename)
                id2img_path[id] = img_path
            else:
                raise ValueError('Could not get patient id from image filename', filename)
        return id2img_path

    def get_landmark_map(self, landmark_path):
        '''
        Parse the landmark csv file to create a dictionary mapping patient ID to landmarks.

        :param landmark_path: Path to landmark file.
        :return: Dictionary mapping patient ID to landmarks.
        '''

        try:
            df = pd.read_excel(landmark_path)
        except ValueError:
            df = pd.read_csv(landmark_path)

        id2landmarks = {}
        for index, row in df.iterrows():
            id, landmarks = self.parse_row(row)
            if id in id2landmarks:
                raise ValueError('Duplicate patient id; one will get overwritten.')
            else:
                id2landmarks[id] = landmarks
        return id2landmarks

    def parse_row(self, row, num_annotations=9):
        '''
        For a given row of data containing annotations for a single patient, we want to extract the landmarks
        into a 2D array with shape (num landmarks, 3 coordinates).

        :param row: Row of data for a single patient.
        :param num_annotations: Number of annotations in the row.
        :return: 2D array with shape (num landmarks, 3 coordinates).
        '''

        id = str(row[0]).lower()
        match_obj = re.match('md(\d\d\d)', id)
        if match_obj:
            id = int(match_obj.group(1))
        else:
            raise ValueError("The following patient identifier is not well formatted: " + id)

        landmarks = []
        pattern = '(\d+),(\d+),(\d+)'
        all_digits = '\d+'
        for i in range(2, 2 + num_annotations):
            data = str(row[i])
            data = re.sub('\s', "", data)
            match_obj = re.match(pattern, data)
            if match_obj:
                x, y, z = match_obj.group(1), match_obj.group(2), match_obj.group(3)
            elif re.match(all_digits, data) and (len(data) == 8 or len(data) == 9):
                # Excel converted text into number: 12,123,123 --> 12123123
                end_nums = 3 if len(data) == 9 else 2
                x, y, z = data[0:end_nums], data[end_nums:end_nums + 3], data[end_nums + 3:end_nums + 6]
            else:
                raise ValueError(f'Patient {id} does not have well formatted data: {data}')

            x, y, z = float(x), float(y), float(z)
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            landmarks.append((x, y, z))

        return id, landmarks


def create_split(all_patient_data, patient_ids, goal_landmark, data_folder, prefix='train'):
    '''
    Creates one training set containing patients from patients_ids and landmark from goal_landmark.

    :param all_patient_data: A list of PatientData objects (all from one type of MRI scan i.e. fat or water)
        that is a superset of patient_ids.
    :param patient_ids: Patients data to use to create this training set.
    :param goal_landmark: The landmark/localization task for this training set.
    :param data_folder: The folder to save this training set to.
    :param prefix: Prefix to use for filenames when creating new files.
    :return:
    '''


    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    image_paths = os.path.join(data_folder, f'{prefix}_image_paths.txt')
    all_landmarks_paths = os.path.join(data_folder, f'{prefix}_landmark_paths.txt')

    image_paths_f = open(image_paths, 'w')
    all_landmarks_f = open(all_landmarks_paths, 'w')

    for patient_data in all_patient_data:
        for patient_id in patient_ids:
            patient_id_str = str(patient_id)
            landmark_path = os.path.join(data_folder, f'{patient_data.filename_gen(patient_id_str)}.txt')
            landmark_path = os.path.abspath(landmark_path)

            landmark_f = open(landmark_path, 'w')
            x, y, z = patient_data.id2landmarks[patient_id][goal_landmark]
            landmark_f.write(f'{x},{y},{z}')
            landmark_f.close()

            all_landmarks_f.write(landmark_path + '\n')
            image_paths_f.write(patient_data.id2img_paths[patient_id] + '\n')

    image_paths_f.close()
    all_landmarks_f.close()


def split_water_fat(all_patient_data):
    '''
    Organize list of PatientData objects into water and fat patient data.

    :param all_patient_data: List of PatientData objects.
    :return: Tuple of 2 lists, one for water PatientData objects and one for fat PatientData objects.
    '''

    water_patient_data = []
    fat_patient_data = []

    for patient_data in all_patient_data:
        if patient_data.scan_type == 'water':
            water_patient_data.append(patient_data)
        elif patient_data.scan_type == 'fat':
            fat_patient_data.append(patient_data)
        else:
            raise ValueError(f'Could not identify type of MRI scan for {patient_data.image_folder}')

    return water_patient_data, fat_patient_data


def split_data(landmark_paths, img_paths, gen_filenames, output_folder, train_split=10, seed=0):
    '''
    This function randomly splits the 30 patients into 10-10-10 groups; the first two groups are
    used for training (for a transfer learning experiment) while the last 10 is used for validation. For
    each of the 10 training patients, we generate 18 training sets, one for each unique pair of
    (anatomical body part, modality). Note that a given patient has 9 anatomical landmark labels and 2 modalities
    (water and fat), so 18 pairs are possible. Each training set has the proper format expected by out DQN for training.
    The validation group has also been formatted and included in the 18*2=36 training sets.

    :param landmark_paths: Path to landmark files, one for each image folder.
    :param img_paths: Paths to folders of images (typically one path for folder of water images and
        one path for folder of fat images)
    :param gen_filenames: Function to generate names for image and label files to be used for training.
    :param output_folder: Folder to store 18*2=36 training sets, which have been formatted for our DQN model.
    :param train_split: Number of patients in train splits (note that there are two train splits)
    :param seed: Random seed.
    '''

    parsed_data_path = os.path.join(output_folder, 'parsed_data')

    if not os.path.isdir(parsed_data_path):
        os.mkdir(parsed_data_path)

    all_patient_data = []
    for landmark_path, img_path, gen_filename in zip(landmark_paths, img_paths, gen_filenames):
        all_patient_data.append(PatientData(landmark_path, img_path, gen_filename))
    water, fat = split_water_fat(all_patient_data)

    num_patients = 30
    indices = np.arange(1, num_patients+1)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices1 = indices[:train_split]
    train_indices2 = indices[train_split:train_split*2]
    val_indices = indices[train_split*2:]

    train_indices1.sort()
    train_indices2.sort()
    val_indices.sort()

    for landmark_goal in range(0, 9):
        # Splitting training data
        data_folder = os.path.join(parsed_data_path, f'set_1_water_landmark_{landmark_goal}')
        create_split(water, train_indices1, landmark_goal, data_folder)

        data_folder = os.path.join(parsed_data_path, f'set_2_water_landmark_{landmark_goal}')
        create_split(water, train_indices2, landmark_goal, data_folder)

        data_folder = os.path.join(parsed_data_path, f'set_1_fat_landmark_{landmark_goal}')
        create_split(fat, train_indices1, landmark_goal, data_folder)

        data_folder = os.path.join(parsed_data_path, f'set_2_fat_landmark_{landmark_goal}')
        create_split(fat, train_indices2, landmark_goal, data_folder)

        # Splitting validation data
        data_folder = os.path.join(parsed_data_path, f'set_1_water_landmark_{landmark_goal}')
        create_split(water, val_indices, landmark_goal, data_folder, prefix='val')

        data_folder = os.path.join(parsed_data_path, f'set_2_water_landmark_{landmark_goal}')
        create_split(water, val_indices, landmark_goal, data_folder, prefix='val')

        data_folder = os.path.join(parsed_data_path, f'set_1_fat_landmark_{landmark_goal}')
        create_split(fat, val_indices, landmark_goal, data_folder, prefix='val')

        data_folder = os.path.join(parsed_data_path, f'set_2_fat_landmark_{landmark_goal}')
        create_split(fat, val_indices, landmark_goal, data_folder, prefix='val')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parses the annotations and images into the correct format for MIDRL')
    parser.add_argument('-i', '--imgs', nargs='+', required=True, help="Path to the folder containing the images")
    parser.add_argument('-l', '--labels', nargs='+', required=True, help="Path to the excel file containing the annotations")
    parser.add_argument('-o', '--output', required=True, help="Path to the folder to store the parsed data")
    parser.add_argument('-c', '--custom_names', nargs='*', help='Custom name templates for the annotation and image files')
    args = parser.parse_args()

    assert len(args.imgs) == len(args.labels)
    if not os.path.isdir(args.output): os.mkdir(args.output)

    gen_filenames = get_filename_generators(args.custom_names, len(args.imgs))
    split_data(args.labels, args.imgs, gen_filenames, args.output)