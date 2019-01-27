import torch
import numpy as np
from torchvision import transforms, utils


# tranforms


class Normalize(object):
    """Normalizes keypoints.
    """

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        ##############################################################
        # TODO: Implemnet the Normalize function, where we normalize #
        # the image from [0, 255] to [0,1] and keypoints from [0, 96]#
        # to [-1, 1]                                                 #
        ##############################################################
        image = self.normalize(1, 0, np.max(image), np.min(image), image)
        key_pts = self.normalize(1, -1, np.max(key_pts), np.min(key_pts), key_pts)
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return {'image': image, 'keypoints': key_pts}

    def normalize(self, newmax, newmin, oldmax, oldmin, value):
        return (newmax - newmin) / (oldmax - oldmin) * (value - oldmax) + newmax


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float()}
