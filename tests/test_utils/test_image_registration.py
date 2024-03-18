import os
import numpy as np
import SimpleITK as sitk

from registration_utils.affine_transforms import RandomZRot
from registration_utils.image_registration import PatientData
from registration_utils.distort_dataset import save_nii_file


def find_affine_with_perfect_landmarks():
    def convert_tensor_to_lis_of_tuple(t):
        dim, num_pts = t.shape
        lis = []
        for i in range(num_pts):
            lis.append((t[0][i], t[1][i], t[2][i]))

        return lis

    fake_img = np.zeros((320, 240, 192))
    rot45 = RandomZRot(45 * np.pi / 180, 45 * np.pi / 180, verbose=True)
    affine_mat, output_shape = rot45.get_affine_mat_output_shape(fake_img)

    np.random.seed(0)
    fake_points = np.random.random((3, 9)) # 9 pts, each with 3 dims
    homo_points = np.vstack([fake_points, np.ones((1, 9))])
    aug_points = affine_mat @ homo_points # Has shape 3 x 9

    a_path = '/raid/home/slai16/research/augs/water_rot45/distorted_images/water_rot45_MD001.nii'
    fake_data = PatientData(convert_tensor_to_lis_of_tuple(fake_points), a_path)
    aug_data = PatientData(convert_tensor_to_lis_of_tuple(aug_points), a_path)
    fake2aug_affine_mat = fake_data.register_landmarks(aug_data)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print('GOAL affine mat')
    print(affine_mat[:3, :])

    print('ESTIMATE affine mat')
    print(fake2aug_affine_mat)


def register_w_perfect_affine_mat():
    end_path = '/home/slai16/research/rl_registration/MD_RL_nii_F_10052021/fat_normal_MD001.nii'
    start_path = '/raid/home/slai16/research/augs/water_rot45/distorted_images/water_rot45_MD001.nii'
    output_path = os.path.join('/home/slai16/Downloads', f'test.nii')

    end_img = sitk.ReadImage(end_path)
    end_img = sitk.GetArrayFromImage(end_img)
    end_img = end_img.transpose((2, 1, 0)) # Has dimensions x, y, z

    rot45 = RandomZRot(45 * np.pi / 180, 45 * np.pi / 180, verbose=True)
    affine_mat, output_shape = rot45.get_affine_mat_output_shape(end_img)

    start = PatientData(None, start_path)
    end = PatientData(None, end_path)

    # To register start image to end image, we need the affine matrix that converts end image to start image
    transformed_img = start.register_image(end, affine_mat)
    transformed_img = sitk.GetImageFromArray(transformed_img)
    save_nii_file(transformed_img, output_path)


if __name__ == '__main__':
    register_w_perfect_affine_mat()
    find_affine_with_perfect_landmarks()
    pass
