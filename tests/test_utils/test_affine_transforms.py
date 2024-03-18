import SimpleITK as sitk
import scipy.ndimage
import numpy as np
import os

from registration_utils import affine_transforms


def get_sample():
    nii_path = '/home/slai16/research/rl_registration/MD_RL_nii_W_11042021/normal_fat_MD001.nii'
    img = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose((2, 1, 0))

    left_right_knee = np.array([[251, 78],
                                [73, 80],
                                [6, 6],
                                [1, 1]])

    return img, left_right_knee


def apply_transform(affine_cls, img, landmarks):
    affine_mat, output_shape = affine_cls.get_affine_mat_output_shape(img)

    inv_affine = np.linalg.inv(affine_mat)
    transformed_img = scipy.ndimage.affine_transform(img, inv_affine, output_shape=output_shape)
    transformed_img = transformed_img.transpose((2, 1, 0))  # has shape z, y, x
    transformed_landmarks = affine_mat @ landmarks
    return transformed_img, transformed_landmarks[:3, :].T


def save_sample(sitk_img, filename):
    test_folder = '/home/slai16/Downloads/distorted_samples'
    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)
    output_path = os.path.join(test_folder, f'{filename}.nii')
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(sitk_img)


def save_test_samples():
    img, landmarks = get_sample()

    rot = affine_transforms.RandomZRot(5 * np.pi / 180, 5 * np.pi / 180, verbose=True)
    transformed_img, _ = apply_transform(rot, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)
    save_sample(transformed_img, '5degree_z_rot')

    trans = affine_transforms.RandomTrans((20, 20), (20, 20), verbose=True)
    transformed_img, _ = apply_transform(trans, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)
    save_sample(transformed_img, '20px_xy_translation')

    scale = affine_transforms.RandomScale((1.25, 1.25), (1.25, 1.25), (1.0, 1.0), verbose=True)
    transformed_img, _ = apply_transform(scale, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)
    save_sample(transformed_img, '1_25_xyscale')

    scale = affine_transforms.RandomScale((1.0, 1.0), (1.0, 1.0), (2.0, 2.0), verbose=True)
    transformed_img, _ = apply_transform(scale, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)
    save_sample(transformed_img, '2x_resample')


def test_RandomZRot():
    img, landmarks = get_sample()
    # Force 20 degree rotation
    rot = affine_transforms.RandomZRot(20 * np.pi / 180, 20 * np.pi / 180, verbose=True)
    transformed_img, transformed_landmarks = apply_transform(rot, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)

    sitk.Show(transformed_img)
    print('Transformed Landmarks:')
    print(transformed_landmarks)


def test_RandomTrans():
    img, landmarks = get_sample()
    # Force 20 translation in x and y
    trans = affine_transforms.RandomTrans((20, 20), (20, 20), verbose=True)
    transformed_img, transformed_landmarks = apply_transform(trans, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)

    sitk.Show(transformed_img)
    print('Transformed Landmarks:')
    print(transformed_landmarks)


def test1_RandomScale():
    img, landmarks = get_sample()
    # Force 1.5 x scale, 1.5 y scale, and 2.0 z scale
    scale = affine_transforms.RandomScale((1.5, 1.5), (1.5, 1.5), (2.0, 2.0), verbose=True)
    transformed_img, transformed_landmarks = apply_transform(scale, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)

    sitk.Show(transformed_img)
    print('Transformed Landmarks:')
    print(transformed_landmarks)


def test2_RandomScale():
    img, landmarks = get_sample()
    # Force 0.75 x scale, 0.75 y scale, and 0.75 z scale
    scale = affine_transforms.RandomScale((0.75, 0.75), (0.75, 0.75), (0.75, 0.75), verbose=True)
    transformed_img, transformed_landmarks = apply_transform(scale, img, landmarks)
    transformed_img = sitk.GetImageFromArray(transformed_img)

    sitk.Show(transformed_img)
    print('Transformed Landmarks:')
    print(transformed_landmarks)


if __name__ == '__main__':
    # test_RandomZRot()
    # test_RandomTrans()
    # test1_RandomScale()
    # test2_RandomScale()
    # save_test_samples()
    pass