from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import pandas as pd
import os
import numpy as np

from exercise_code.data_utils import get_image, get_keypoints
from exercise_code.vis_utils import show_all_keypoints


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            custom_point (list): which points to train on
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame.dropna(inplace=True)
        self.key_pts_frame.reset_index(drop=True, inplace=True)
        self.transform = transform

    def __len__(self):
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataset                                     #
        ########################################################################
        return self.key_pts_frame.shape[0]
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def __getitem__(self, idx):
        ########################################################################
        # TODO:                                                                #
        # Return the idx sample in the Dataset. A simple should be a dictionary#
        # where the key, value should be like                                  #
        #        {'image': image of shape [C, H, W],                           #
        #         'keypoints': keypoints of shape [num_keypoints, 2]}          #
        ########################################################################
        image = get_image(idx, self.key_pts_frame)
        image = image.reshape(1, image.shape[0], image.shape[1])
        keypoints = get_keypoints(idx, self.key_pts_frame)
        sample = {'image': image,
                  'keypoints': keypoints}
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        if self.transform:
            sample = self.transform(sample)

        return sample


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# face_dataset = FacialKeypointsDataset(csv_file='/Users/w.jurasz/Studies/DL/i2dl/exercise_4/datasets/training.csv')
#
# num_to_display = 3
#
# for i in range(num_to_display):
#     # define the size of images
#     fig = plt.figure(figsize=(20, 10))
#
#     # randomly select a sample
#     rand_i = np.random.randint(0, len(face_dataset))
#     sample = face_dataset[rand_i]
#
#     # print the shape of the image and keypoints
#     print('index: {}'.format(i))
#     print('image size: {}'.format(sample['image'].shape))
#     print('keypoint shape: {}'.format(sample['keypoints'].shape))
#
#     ax = plt.subplot(1, num_to_display, i + 1)
#     ax.set_title('Sample #{}'.format(i))
#
#     # Using the same display function, defined earlier
#     show_all_keypoints(sample['image'][0], sample['keypoints'])
