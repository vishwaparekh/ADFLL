'''
This file uses affine transformations defined in affine_transformations.py to provide offline 
augmentations. These augmented images are to improve robustness of localization models. 

Author: Shuhao Lai <shuhaolai18@gmail.com>
'''


import SimpleITK as sitk
import scipy.ndimage
import pandas as pd
import numpy as np
import warnings
import os
import re

from registration_utils.preprocess_data import get_num
from registration_utils.affine_transforms import RandomZRot, RandomTrans, RandomScale, RandAug


def parse_annotations(annotations_path):
    '''
    Extracts all landmark locations from a given csv file and stores the landmarks 
    in a 3D list with shape (patient, landmark, xyORz).  
    
    :param annotations_path: Path to landmark csv.
    :return: 3D list of parsed landmarks 
    '''
    df = pd.read_excel(annotations_path)
    all_landmarks = []
    for index, row in df.iterrows():
        landmarks = parse_row(row)
        all_landmarks.append(landmarks)
    return all_landmarks


def parse_row(row, num_annotations=9):
    '''
    Parse a row of information, which represents landmark locations for a single patient.

    :param row: Row of annotation data. 
    :param num_annotations: Number of annotations in the row.
    :return: A 2D matrix with shape (3 coordinates, num annotations)
    '''
    
    name = str(row[0])
    pattern = '(\d+),(\d+),(\d+)'
    all_digits = '\d+'
    landmarks = []
    for i in range(2, 2 + num_annotations):
        data = str(row[i])
        data = re.sub('\s', "", data)
        match_obj = re.match(pattern, data)
        if match_obj:
            x, y, z = match_obj.group(1), match_obj.group(2), match_obj.group(3)
        elif re.match(all_digits, data) and (len(data) == 8 or len(data) == 9):
            # Excel converted text into number: 12,123,123 --> 12123123
            end_nums = 3 if len(data) == 9 else 2
            x, y, z = data[0:end_nums], data[end_nums:end_nums+3], data[end_nums+3:end_nums+6]
        else:
            warnings.warn(f'Patient identifier {name} does not have well formatted data: {data}')
            continue
        landmark = np.array([[int(x)],
                             [int(y)],
                             [int(z)]], dtype=np.uint)
        landmarks.append(landmark)
    landmarks = np.hstack(landmarks)
    return landmarks


def landmarks2str(landmarks):
    '''
    Converts the landmarks to a string that can be saved as a csv file. 
    :param landmarks: Landmarks matrix with shape (3 x n)
    :return: Return string of landmarks, ready to be saved as csv file. 
    '''
    
    def landmark2str(landmark):
        return f'"{int(landmark[0])}, {int(landmark[1])}, {int(landmark[2])}"'

    ndims, nlabels = landmarks.shape
    landmark_str = landmark2str(landmarks[:, 0])
    for i in range(1, nlabels):
        landmark = landmarks[:, i]
        landmark_str += ',' + landmark2str(landmark)
    return landmark_str


def transform_nii(nii_path, affine_cls, landmarks=None):
    '''
    Applies the provided affine transformation object (which must already be initialized with correct parameters)
    to the image from the provided path. 
    
    :param nii_path: Path to image in nii format. 
    :param affine_cls: Affine transformation object. 
    :param landmarks: If provided, landmarks are also transformed. 
    :return: Transformed image and, if provided, transformed landmarks.
    '''
    
    # Reading in image
    img = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose((2, 1, 0))  # has shape x, y, z now insted of z, y, x

    affine_mat, output_shape = affine_cls.get_affine_mat_output_shape(img)
    inv_affine = np.linalg.inv(affine_mat)
    transformed_img = scipy.ndimage.affine_transform(img, inv_affine, output_shape=output_shape)
    transformed_img = transformed_img.transpose((2, 1, 0))  # has shape z, y, x

    if landmarks is not None:
        # Transforming landmarks
        row_of_ones = np.ones((1, landmarks.shape[1]))
        homogeneous_points = np.vstack([landmarks, row_of_ones])
        new_landmarks = affine_mat @ homogeneous_points
        new_landmarks = new_landmarks[:3, :]

        # Checking if transformed landmarks are within bounds
        img_z, img_y, img_x = transformed_img.shape
        x = np.round(new_landmarks[0, :])
        y = np.round(new_landmarks[1, :])
        z = np.round(new_landmarks[2, :])
        if np.any(x < 0) or np.any(y < 0) or np.any(z < 0) or np.any(x >= img_x) or np.any(y >= img_y) or np.any(z >= img_z):
            warnings.warn(f'Landmark was transformed out of bounds for image {nii_path}')

        return transformed_img, new_landmarks
    else:
        return transformed_img


def resample_nii(nii_path, landmarks=None, zscale=None, z=None):
    '''
    Resamples (scale the z-axis) for a 3D image using an inverse affine transformation matrix to select 
    pixels from the input that are used to generate a given output pixel. 
    
    :param nii_path: The path to the 3D image. 
    :param landmarks: If provided, the landmarks will be resampled as well. 
    :param zscale: If provided, the scaling factor for the z axis. 
    :param z: If provided, the number of z frames to generate (the zscale is automatically calculated).
    :return: The transformed image (and landmark if provided). 
    
    Note that only zscale or z need to be provided; if both are provided, zscale is used. 
    '''
    
    # Reading in image
    img = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(img)
    img_z, img_h, img_w = img.shape

    # Setting up transformation matrix
    if zscale:
        z = int(img_z * zscale)
        if z != img_z * zscale:
            # Scaled z is not an integer
            z = int(round(z))
            zscale = z / img_z
    elif z:
        # Checking that z is an integer
        assert z == int(z)
        z = int(z)
        zscale = z / img_z
    else:
        raise ValueError('Please provide zscale or z argument to specify how to scale image')

    # Note that the diagonals for the transformation matrix scales dim0, dim1, dim2
    # Thus, to transform the z dimension, we need to set zscale in top left corner
    zscale_mat = np.array([[zscale, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    inv_zscale_mat = np.linalg.inv(zscale_mat)

    # Applying transformation to resample image
    output_shape = (z, img_h, img_w)
    # Uses spline interpolation to the third order (by default)
    transformed_img = scipy.ndimage.affine_transform(img, inv_zscale_mat, output_shape=output_shape)

    if landmarks is not None:
        # Transforming landmarks
        # Note that z is in first row to align with zscale_mat
        landmarks[[2, 0]] = landmarks[[0, 2]]
        row_of_ones = np.ones((1, landmarks.shape[1]))
        homogeneous_points = np.vstack([landmarks, row_of_ones])
        new_landmarks = zscale_mat @ homogeneous_points
        new_landmarks = new_landmarks[:3, :]
        new_landmarks[[2, 0]] = new_landmarks[[0, 2]]

        return transformed_img, new_landmarks
    else:
        return transformed_img


def visualize_resample_distortion(nii_path, zscale=2.0):
    '''
    Debugger function to resample and visualize an image using SITK.
    
    :param nii_path: Path to image in nii format.
    :param zscale: Scaling factor for the z axis. 
    '''
    
    before_img = sitk.ReadImage(nii_path)
    after_img = resample_nii(nii_path, zscale)
    after_img = sitk.GetImageFromArray(after_img)

    sitk.Show(before_img, title='Before Resample')
    sitk.Show(after_img, title='After Resample')


def save_nii_file(sitk_img, output_path):
    '''
    Helper function to save image (as an SITK instance) to provided path. 
    
    :param sitk_img: Image from SITK instance.
    :param output_path: Path to save image. 
    '''
    
    assert output_path.split('.')[1].strip() == 'nii'

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(sitk_img)


def distorted_w_filename(id):
    return f'resample_W_MRI_{id}'


def basic_filename(id):
    '''
    Function to generate filenames for an MRI scan given an scan ID. 
    :param id: Scan ID.
    :return: Generated filename. 
    '''
    
    return f'aug_MRI_{id}'


def w_filename_func(id):
    '''
    Function to generate filenames for water scans given an scan ID. 
    :param id: Scan ID.
    :return: Filename for a water scan. 
    '''
    
    return f'aug_W_MRI_{id}'


def f_filename_func(id):
    '''
    Function to generate filenames for fat scans given an scan ID. 
    :param id: Scan ID.
    :return: Filename for a fat scan. 
    '''
    
    return f'aug_F_MRI_{id}'


def resample_and_save(nii_folder, landmark_file, output_folder):
    '''
    Resamples all images to have twice as many frames. Landmarks are resampled as well.
    All outputs are saved to specified folder.
    
    :param nii_folder: Folder to find nii images.
    :param landmark_file: Path to csv containing landmark labels. 
    :param output_folder: Folder to saved resampled outputs.
    '''
    
    output_nii_folder = os.path.join(output_folder, 'resampled_nii_W')
    if not os.path.isdir(output_nii_folder):
        os.mkdir(output_nii_folder)

    all_landmarks = parse_annotations(landmark_file)
    new_landmark_path = os.path.join(output_folder, f'distorted_landmarks.csv')
    f = open(new_landmark_path, "w")
    headers = 'Patient Name, Date, L. Knee, R. Knee, L. Trochanter, R. Trochanter, L. Kidney, R. Kidney, Spleen, L. Lung, R.Lung'
    f.write(headers + '\n')

    filenames = os.listdir(nii_folder)
    filenames.sort(key=get_num)
    id_pattern = 'MD(\d\d\d)'

    for filename in filenames:
        print(f'Processing {filename}')
        match_obj = re.search(id_pattern, filename)
        patient_name = match_obj.group()
        patient_num = int(match_obj.group(1))

        new_filename = distorted_w_filename(match_obj.group())
        input_path = os.path.join(nii_folder, filename)
        output_path = os.path.join(output_nii_folder, f'{new_filename}.nii')

        patient_landmarks = all_landmarks[patient_num - 1]
        distorted_img, distorted_landmarks = resample_nii(input_path, patient_landmarks, zscale=2)
        landmarks_str = landmarks2str(distorted_landmarks)

        distorted_img = sitk.GetImageFromArray(distorted_img)
        save_nii_file(distorted_img, output_path)
        f.write(f'{patient_name},NoDate,{landmarks_str}\n')  # Must include patient name and date for next preprocess step.
    f.close()


def distort_and_save(nii_folder, landmark_file, output_folder, affine_cls, filename_func=basic_filename):
    '''
    Helper function to distort the provided MRI scans and landmarks using a provided transformation object affine_cls. 
    The augmented images and labels are saved to the specified folder. 
    
    :param nii_folder: Folder containing MRI scans in nii format.
    :param landmark_file: The excel file containing the landmark labels.
    :param output_folder: The folder to save the augmented images and landmarks. 
    :param affine_cls: The initialized object that performs the augmentation (from affine_transforms.py)
    :param filename_func: The function that generates a filename given a scan ID.
    '''

    output_nii_folder = os.path.join(output_folder, 'distorted_images')
    if not os.path.isdir(output_nii_folder):
        os.makedirs(output_nii_folder)

    all_landmarks = parse_annotations(landmark_file)
    new_landmark_path = os.path.join(output_folder, f'distorted_landmarks.csv')
    f = open(new_landmark_path, "w")
    headers = 'Patient Name, Date, L. Knee, R. Knee, L. Trochanter, R. Trochanter, L. Kidney, R. Kidney, Spleen, L. Lung, R.Lung'
    f.write(headers + '\n')

    filenames = os.listdir(nii_folder)
    filenames.sort(key=get_num)
    id_pattern = 'MD(\d\d\d)'

    for filename in filenames:
        print(f'Processing {filename}')
        match_obj = re.search(id_pattern, filename)
        patient_name = match_obj.group()
        patient_num = int(match_obj.group(1))

        new_filename = filename_func(match_obj.group())
        input_path = os.path.join(nii_folder, filename)
        output_path = os.path.join(output_nii_folder, f'{new_filename}.nii')

        patient_landmarks = all_landmarks[patient_num - 1]
        distorted_img, distorted_landmarks = transform_nii(input_path, affine_cls=affine_cls, landmarks=patient_landmarks)
        landmarks_str = landmarks2str(distorted_landmarks)

        distorted_img = sitk.GetImageFromArray(distorted_img)
        save_nii_file(distorted_img, output_path)
        f.write(f'{patient_name},NoDate,{landmarks_str}\n')  # Must include patient name and date for next preprocess step.
    f.close()


def resample_water():
    '''
    This function resamples the water MRI scans and landmarks and saved them to a specified folder. 
    '''
    
    nii_folder = '/home/slai16/research/rl_registration/MD_RL_nii_W_11042021'
    landmark_file = '/home/slai16/research/rl_registration/RL_landmarks_10052021.xlsx'
    output_folder = '/home/slai16/research/rl_registration'
    resample_and_save(nii_folder, landmark_file, output_folder)


def set_offline_augs():
    '''
    For each fat and water MRI scan, this function applies the following 3D transformations and saves the output 
    in the specified folder:
    
        (1) Rotation along z axis by 45 degrees
        (2) Rotation along z axis by -45 degrees
        (3) Scale xy by 1/2 
        (4) Scale xy by 1.5
        (5) Resample to 2x frames
        (6) Resample to 1/2x frames

    The transformations are applied to the landmark labels and saved as well. Note that this 
    function applies the transformations deterministically (no randomness as in rand_offline_augs()). 
    '''
    
    rot45 = RandomZRot(45 * np.pi / 180, 45 * np.pi / 180, verbose=True)
    rot_neg45 = RandomZRot(-45 * np.pi / 180, -45 * np.pi / 180, verbose=True)
    scale0_5 = RandomScale((0.5, 0.5), (1.0, 1.0), verbose=True)
    scale1_5 = RandomScale((1.5, 1.5), (1.0, 1.0), verbose=True)
    resample2_0 = RandomScale((1.0, 1.0), (2.0, 2.0), verbose=True)
    resample0_5 = RandomScale((1.0, 1.0), (0.5, 0.5), verbose=True)
    transforms_list = [rot45, rot_neg45, scale0_5, scale1_5, resample2_0, resample0_5]
    transform_names_list = ['rot45', 'rot_neg45', 'scale0_5', 'scale1_5', 'resample2_0', 'resample0_5']

    w_nii_folder = '/home/slai16/research/rl_registration/MD_RL_nii_W_11042021'
    f_nii_folder = '/home/slai16/research/rl_registration/MD_RL_nii_F_10052021'
    landmark_file = '/home/slai16/research/rl_registration/RL_landmarks_10052021.xlsx'
    output_folder_base = '/raid/home/slai16/research/augs'

    for transform, transform_name in zip(transforms_list, transform_names_list):
        w_output_folder = os.path.join(output_folder_base, f'water_{transform_name}')
        f_output_folder = os.path.join(output_folder_base, f'fat_{transform_name}')

        print(f'PROCESSING TRANSFORMATION {transform_name}')
        print('PROCESSING WATER SCANS')
        distort_and_save(w_nii_folder, landmark_file, w_output_folder, transform, w_filename_func)

        print('PROCESSING FAT SCANS')
        distort_and_save(f_nii_folder, landmark_file, f_output_folder, transform, f_filename_func)


def rand_offline_augs():
    '''
    Applies random transformations on water and fat MRI scans and their corresponding landmark labels.
    The random transformations are uniformly sampled from a list of provided 3D affine transformations 
    (please view affine_transforms.py for more details on the transformations). These offline transformations are 
    used to train a more robust landmark localization model. These transformed images are saved in 
    specified output_folder_base file. 
    '''
    
    transforms_list = [
        RandomZRot(-5 * np.pi / 180, 5 * np.pi / 180, verbose=True),
        RandomTrans((-15, 15), (-15, 15), verbose=True),
        RandomScale((0.75, 1.25), (1.0, 1.0), verbose=True),
        RandomScale((1.0, 1.0), (0.5, 2.0), verbose=True)
    ]
    rand_aug = RandAug(transforms_list)

    w_nii_folder = '/home/slai16/research/rl_registration/MD_RL_nii_W_11042021'
    f_nii_folder = '/home/slai16/research/rl_registration/MD_RL_nii_F_10052021'
    landmark_file = '/home/slai16/research/rl_registration/RL_landmarks_10052021.xlsx'
    output_folder_base = '/home/slai16/research/rl_registration/augs'

    water_folder_base = os.path.join(output_folder_base, 'water')
    fat_folder_base = os.path.join(output_folder_base, 'fat')
    num_iters = 3

    for i in range(num_iters):
        w_output_folder = os.path.join(water_folder_base, f'aug_set_{i}')
        f_output_folder = os.path.join(fat_folder_base, f'aug_set_{i}')

        print(f'ITERATION {i}')
        print('PROCESSING WATER SCANS')
        distort_and_save(w_nii_folder, landmark_file, w_output_folder, rand_aug, w_filename_func)

        print('PROCESSING FAT SCANS')
        distort_and_save(f_nii_folder, landmark_file, f_output_folder, rand_aug, f_filename_func)


if __name__ == '__main__':
    set_offline_augs()
