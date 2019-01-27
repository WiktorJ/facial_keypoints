import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from exercise_code.dataloader import FacialKeypointsDataset


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.resnet = models.inception_v3(True)
        for name, param in self.resnet.named_parameters():
            if "7c" not in name and "fc" not in name:
                param.requires_grad = False
        self.resnet.fc = nn.Linear(2048, 1000)
        # self.resnet.transform_input = False
        self.resnet.aux_logits = False
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(1000, 30)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = F.upsample(x, size=(299, 299))
        x1 = self.resnet(x)
        x2 = self.relu(x1)
        x3 = self.dropout(x2)
        x4 = self.fc(x3)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x4

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


#

# import matplotlib.pyplot as plt
# from exercise_code.vis_utils import show_all_keypoints
#
# from torch.utils.data import DataLoader
#
# import torch
# from torchvision import transforms, utils
# # tranforms
#
# from exercise_code.transforms import Normalize, ToTensor
# from torchvision.transforms import Resize
#
# data_transform = transforms.Compose([Normalize(),
#                                      ToTensor()])
# transformed_dataset = FacialKeypointsDataset(
#     csv_file='/Users/w.jurasz/Studies/DL/i2dl/exercise_4/datasets/training.csv',
#     transform=data_transform)
# batch_size = 20
# train_loader = DataLoader(transformed_dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=4)
#
# model = KeypointModel()
#
#
# def show_sample_outputs(image, keypoints, i):
#     # define the size of images
#     fig = plt.figure(figsize=(20, 20))
#
#     # Predict with model
#     predicted_keypoints = model(image)[0]
#     # Cast back to (x,y)-coordinates
#     predicted_keypoints = predicted_keypoints.view(-1, 2).detach()
#
#     # Undo data normalization
#     image = torch.squeeze(image) * 255.
#     keypoints = (keypoints.view(-1, 2) * 48) + 48
#     predicted_keypoints = (predicted_keypoints * 48) + 48
#
#     # print the shape of the image and keypoints
#     print('index: {}'.format(i))
#     print('image shape: {}'.format(image.shape))
#     print('gt keypoints shape: {}'.format(keypoints.shape))
#     print('predict keypoints shape: {}'.format(predicted_keypoints.shape))
#
#     # Print data loader image
#     ax = plt.subplot(4, 1, 1)
#     ax.set_title('Sample #{}: Dataloader'.format(i))
#     # Using the same display function, defined earlier
#     image = image[1]
#     show_all_keypoints(image, keypoints)
#
#     # Print predicted image
#     ax = plt.subplot(1, 1, 1)
#     ax.set_title('Sample #{}: Prediction'.format(i))
#     # Using the same display function, defined earlier
#     show_all_keypoints(image, predicted_keypoints)
#     plt.show()
#
#
# num_to_display = 3
# for idx, sample in enumerate(train_loader):
#     if idx == num_to_display:
#         break
#     image2c = torch.cat((sample['image'][0:1], sample['image'][0:1]))
#     image3c = torch.cat((image2c, sample['image'][0:1])).view(1, 3, 96, 96)
#     show_sample_outputs(image3c, sample['keypoints'][0:1], idx)
