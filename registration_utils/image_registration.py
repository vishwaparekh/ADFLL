import os
import re
import cv2
import warnings
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import registration_utils.distort_dataset as dd

from collections import defaultdict
from registration_utils.eval_pred_diff_modalities import parse_data_line


class PatientData:
    def __init__(self, landmarks, image_path):
        self.landmarks = landmarks
        self.image_path = image_path
        self.image_name = os.path.basename(image_path).split('.')[0] # Excludes nii extension
        self.LANDMARKS2USE = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    @property
    def img_size(self):
        img = sitk.ReadImage(self.image_path)
        img = sitk.GetArrayFromImage(img)
        img_z, img_y, img_x = img.shape
        return img_x, img_y, img_z

    def register_landmarks(self, end_patient_data):
        '''
        Estimates affine transformation matrix that transforms self.landmarks to end_patient_data.landmarks
        '''

        start_points_set = np.zeros((9, 3))
        end_points_set = np.zeros((9, 3))
        for agent_i in self.LANDMARKS2USE:
            start_points = self.landmarks[agent_i]
            end_points = end_patient_data.landmarks[agent_i]
            start_points_set[agent_i] = start_points
            end_points_set[agent_i] = end_points
        retval, affine_mat, inliers = cv2.estimateAffine3D(start_points_set, end_points_set)
        return affine_mat

    def register_image(self, end_patient_data, affine_mat=None):
        if affine_mat is None:
            affine_mat = self.register_landmarks(end_patient_data)

        img = sitk.ReadImage(self.image_path)
        img = sitk.GetArrayFromImage(img)
        img = img.transpose((2, 1, 0))

        output_shape = end_patient_data.img_size
        transformed_img = scipy.ndimage.affine_transform(img, affine_mat, output_shape=output_shape)
        transformed_img = transformed_img.transpose((2, 1, 0))  # z, y, x as expected by sitk
        return transformed_img


def parse_best_landmark_file(path):
    '''
    Converts text file of best landmarks for each agent into a dictionary where image filename is the key
    and the value is a list of landmarks.
    '''

    with open(path) as file:
        NUM_AGENTS = 9
        all_landmarks = {}
        for line in file:
            filename = line.lower().strip()
            if filename in all_landmarks:
                raise ValueError('Two or more sets of landmarks for the same image.')

            img_landmarks = []
            for i in range(NUM_AGENTS):
                loc = parse_data_line(next(file))
                img_landmarks.append(loc)
            all_landmarks[filename] = img_landmarks

        return all_landmarks


def get_img_name2img_path(all_img_paths_file):
    with open(all_img_paths_file) as f:
        img_name2img_path = dict()
        for img_path in f:
            img_path = img_path.strip()
            if len(img_path) > 0:
                img_name = os.path.basename(img_path)
                img_name = img_name.lower().strip()
                img_name2img_path[img_name] = img_path
        return img_name2img_path


def group_landmarks(best_landmark_path, all_img_paths_file, src_pattern):
    '''
    Aggregate landmarks for the same patient. The output data will have the following structure:

    {
        'patient_id' : {
            'src': src_PatientData
            'others': [
                other1_PatientData,
                other2_PatientData, ...
            ]
        }, ...
    }
    '''

    img_name2img_path = get_img_name2img_path(all_img_paths_file)
    all_landmarks = parse_best_landmark_file(best_landmark_path)
    id_pattern = 'md\d\d\d'
    agg_patients_landmarks = defaultdict(lambda: {})
    for img_name, landmarks in all_landmarks.items():
        img_name = img_name.lower().strip()
        img_path = img_name2img_path[img_name]

        match_obj = re.search(id_pattern, img_name)
        id = match_obj.group()

        if re.search(src_pattern, img_name):
            # Landmarks for this image is the source; other landmarks from the same patient
            # should be registered to this image
            if 'src' in agg_patients_landmarks[id]:
                raise ValueError('For a given patient, there are two or more src landmarks')
            else:
                agg_patients_landmarks[id]['src'] = PatientData(landmarks, img_path)
        else:
            if 'others' not in agg_patients_landmarks[id]:
                agg_patients_landmarks[id]['others'] = []

            agg_patients_landmarks[id]['others'].append(PatientData(landmarks, img_path))

    return agg_patients_landmarks


def register_all_imgs(best_landmarks_path, all_img_paths_file, output_folder):
    image_folder = os.path.join(output_folder, 'images')
    if not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    src_pattern = 'fat_normal'
    patients_landmarks = group_landmarks(best_landmarks_path, all_img_paths_file, src_pattern)
    patient_landmarks_lis = list(patients_landmarks.items())
    patient_landmarks_lis.sort(key=lambda item: item[0]) # Sort by patient id

    # Make np arrays look nice when printing
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    affine_mat_txt_path = os.path.join(output_folder, 'affine_mats.txt')
    affine_mat_file = open(affine_mat_txt_path, 'w')

    for patient_id, patient_data in patient_landmarks_lis:
        print(f'PROCESSING PATIENT {patient_id}')
        patient_image_folder = os.path.join(image_folder, patient_id)
        if not os.path.isdir(patient_image_folder):
            os.mkdir(patient_image_folder)
        if 'src' in patient_data and 'others' in patient_data:
            src_patient_data = patient_data['src']
            others_patient_data = patient_data['others']
            for other in others_patient_data:
                src2other_affine_mat = src_patient_data.register_landmarks(other)
                other2src_registered_img = other.register_image(src_patient_data, src2other_affine_mat)
                other2src_registered_img = sitk.GetImageFromArray(other2src_registered_img)

                str_mat_descp = f'Affine mat from {src_patient_data.image_name} to {other.image_name}'
                print(str_mat_descp, file=affine_mat_file)
                print(src2other_affine_mat, file=affine_mat_file)

                str_procedure = f'{other.image_name}_registered2_{src_patient_data.image_name}'
                img_output_path = os.path.join(patient_image_folder, f'{str_procedure}.nii')
                dd.save_nii_file(other2src_registered_img, img_output_path)

                print(f'FINISHED PROCESSING {str_procedure}')
        else:
            warnings.warn(f'Patient {patient_id} does not have \'src\' or \'others\' landmarks, so registration cannot be performed')

    affine_mat_file.close()


if __name__ == '__main__':
    best_landmarks_path = '/home/slai16/PycharmProjects/rl_medical_env_2021/MIDRL-registration/train_log/aug_model_best_landmarks_all_aug_w_data.txt'
    all_img_paths_file = '/home/slai16/research/rl_registration/parsed_deterministic_aug_data/val_image_paths_aug_w_norm_f.txt'
    output_folder = '/home/slai16/research/rl_registration/aug_water_norm_f_model_registered_aug_w_img'
    register_all_imgs(best_landmarks_path, all_img_paths_file, output_folder)