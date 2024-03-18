'''
This file provides 3D affine transformation classes wholes transformation parameters
are uniformly sampled (stochastic). These transformations are useful to generate 3D augmented training data.   

Author: Shuhao Lai <shuhaolai18@gmail.com>
'''

from abc import ABC, abstractmethod
import numpy as np


class TransformBase(ABC):
    '''
    This is an abstract class intended to be inherited by other 3D affine transformation class.
    '''
    
    @abstractmethod
    def get_affine_mat_output_shape(self, img):
        '''
        Applies a 3D affine transformation on the image img. 
        
        :param img: The image to apply the transformation to. 
        :return: The affine transformed image.
        '''
        pass


class RandomZRot(TransformBase):
    '''
    Apply a random rotation on the z axis, which is located in the center of the image. 
    '''
    
    def __init__(self, min_rad, max_rad, verbose=False):
        '''
        The rotation angle is uniformly sampled between min_rad and max_rad. 
        Set min_rad = max_rad to deterministically select a rotation angle. 
        
        :param min_rad: The minimum rotation angle in radians. 
        :param max_rad: The maximum rotation anlge in radians.
        :param verbose: True to print the sampled rotation angle. 
        '''
        
        self.min_rad = min_rad
        self.max_rad = max_rad
        self.verbose = verbose

    def get_affine_mat_output_shape(self, img):
        '''
        Apply the random rotation on the z axis on image img. 
        
        :param img: The image to rotate. 
        :return: The rotated image. 
        '''
        
        x, y, z = img.shape
        center = np.array([[1, 0, 0, -int(x/2)],
                           [0, 1, 0, -int(y/2)],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        re_loc = np.array([[1, 0, 0, int(x/2)],
                           [0, 1, 0, int(y/2)],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Rotation around z axis
        theta = np.random.uniform(self.min_rad, self.max_rad)
        rotation_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
        translation_vec = np.array([[0],
                                    [0],
                                    [0]])
        homogenous_row = np.array([[0, 0, 0, 1]])
        affine_rotation = np.hstack([rotation_mat, translation_vec])
        affine_rotation = np.vstack([affine_rotation, homogenous_row])

        # A center rotation
        affine_mat = re_loc @ affine_rotation @ center

        if self.verbose:
            print(f'Rotation: Theta {theta}')

        return affine_mat, img.shape


class RandomTrans(TransformBase):
    '''
    Applies a random translation only on the x and y axis. 
    '''
    
    def __init__(self, x_rang, y_rand, verbose=False):
        '''
        The translation magnitude for x and y are uniform random samples 
        between x_rang for x axis and between y_rang for y axis. 
        
        :param x_rang: A tuple containing (min x translation, max x translation)
        :param y_rand: A tuple containing (min y translation, max y translation)
        :param verbose: True to print sampled translation magnitudes along x and y axes. 
        '''
        self.x_min, self.x_max = x_rang
        self.y_min, self.y_max = y_rand
        self.verbose = verbose

    def get_affine_mat_output_shape(self, img):
        '''
        Apply random translation on provided image. 
        
        :param img: Image to translate. 
        :return: Translated image. 
        '''
        x_trans = np.random.uniform(self.x_min, self.x_max)
        x_trans = int(round(x_trans))

        y_trans = np.random.uniform(self.y_min, self.y_max)
        y_trans = int(round(y_trans))

        trans = np.array([[1, 0, 0, x_trans],
                          [0, 1, 0, y_trans],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        if self.verbose:
            print(f'Translation: x {x_trans}, y {y_trans}')

        return trans, img.shape


class RandomScale(TransformBase):
    '''
    Scaling along the x and y axes are done from the center of the image and do not increase the xy dimensions 
    (giving the impression that the image is getting larger or smaller). Scaling along the z axis allows 
    the number of frames to increase (similar to resampling the 3D image along z axis)
    '''

    def __init__(self, xyscale_range, zscale_range, verbose=False):
        '''
        The scaling factor for the x dimension is equal to that of the y dimension to maintain aspect
        ratio. This scaling factor is drawn from a uniform random distribution, as is the range for the z axis
        scaling factor. 
        
        :param xyscale_range: A tuple representing (min xy scaling factor, max xy scaling factor)
        :param zscale_range: A tuple representing (min z scaling factor, max z scaling factor)
        :param verbose: True to print sampled xy scaling factor and z scaling factor. 
        '''
        # Must scale x and y by the same amount to maintain aspect ratio
        self.xyscale_min, self.xyscale_max = xyscale_range
        self.zscale_min, self.zscale_max = zscale_range
        self.verbose = verbose

    def get_affine_mat_output_shape(self, img):
        '''
        Applies scaling transformation on img. 
        
        :param img: The image to transform. 
        :return: The transformed image. 
        '''
        
        img_x, img_y, img_z = img.shape
        xyscale = np.random.uniform(self.xyscale_min, self.xyscale_max)
        zscale = np.random.uniform(self.zscale_min, self.zscale_max)

        output_z = int(img_z * zscale)
        if output_z != img_z * zscale:
            output_z = int(round(output_z))
        zscale = output_z / img_z

        # Scaled parts exceeding original img size are left out, so must scale from the center
        # No need to center scale the z plane since all z frames will be used
        center = np.array([[1, 0, 0, -int(img_x/2)],
                           [0, 1, 0, -int(img_y/2)],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        re_loc = np.array([[1, 0, 0, int(img_x/2)],
                           [0, 1, 0, int(img_y/2)],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        scale_mat = np.array([[xyscale, 0, 0, 0],
                              [0, xyscale, 0, 0],
                              [0, 0, zscale, 0],
                              [0, 0, 0, 1]])
        affine_mat = re_loc @ scale_mat @ center

        if self.verbose:
            print(f'Scale: xy-scale {xyscale}, z-scale {zscale}')

        return affine_mat, (img_x, img_y, output_z)


class RandAug(TransformBase):
    '''
    Applies a randomly sampled transformation (from a provided list) to an image.
    '''
    def __init__(self, transforms):
        '''
        Generate a list of transformation to sample from.
        
        :param transforms: The list of possible transformations.
        '''
        
        for t in transforms:
            assert isinstance(t, TransformBase)
        self.transforms = transforms

    def get_affine_mat_output_shape(self, img):
        '''
        Randomly selects and applies a transformation from the provided list.
        
        :param img: Image to trasnform.
        :return: Transformed image. 
        '''
        
        t = np.random.choice(self.transforms)
        return t.get_affine_mat_output_shape(img)