## import libraries
import os
import os.path
import numpy as np
import scipy.io as sio

import torch

from PIL import Image
from torchvision import transforms


## define dataset
class Dataset(torch.utils.data.Dataset):
    # dataset preprocessing
    def __init__(self, data_dir, split, imSize=(224,224), gridSize=(25,25)):
        self.data_dir = data_dir
        self.split = split
        self.imSize = imSize
        self.gridSize = gridSize

        metaFile = os.path.join(data_dir, 'metadata.mat')
        self.metadata = sio.loadmat(metaFile, squeeze_me=True, struct_as_record=False)
        self.faceMean = get_mean_image(data_dir=self.data_dir, file_name='mean_face_224.mat')
        self.eyeLeftMean = get_mean_image(data_dir=self.data_dir,file_name='mean_left_224.mat')
        self.eyeRightMean = get_mean_image(data_dir=self.data_dir,file_name='mean_right_224.mat')

        self.transformFace = transforms.Compose(
            [ToNumpy(), ToPILImage(), transforms.Resize(self.imSize), SubtractMean(meanImg=self.faceMean), ToTensor()])
        self.transformEyeL = transforms.Compose(
            [ToNumpy(), ToPILImage(), transforms.Resize(self.imSize), SubtractMean(meanImg=self.eyeLeftMean), ToTensor()])
        self.transformEyeR = transforms.Compose(
            [ToNumpy(), ToPILImage(), transforms.Resize(self.imSize), SubtractMean(meanImg=self.eyeRightMean), ToTensor()])

        if split == 'train':
             mask = self.metadata['labelTrain']
        if split == 'val':
             mask = self.metadata['labelVal']
        if split == 'test':
             mask = self.metadata['labelTest']

        self.indices = np.argwhere(mask)[:, 0]

     # dataset length
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]

        # input (dictionary type only)
        imFace_dir = os.path.join(self.data_dir, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeL_dir = os.path.join(self.data_dir, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeR_dir = os.path.join(self.data_dir, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

        imFace = Image.open(imFace_dir).convert('RGB')
        imEyeL = Image.open(imEyeL_dir).convert('RGB')
        imEyeR = Image.open(imEyeR_dir).convert('RGB')

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        faceGrid = makeGrid(gridSize=self.gridSize, params=self.metadata['labelFaceGrid'][index, :])
        faceGrid = torch.FloatTensor(faceGrid)

        # label
        label = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)
        label = torch.FloatTensor(label)

        # index
        index = torch.LongTensor([int(index)])

        return imFace, imEyeL, imEyeR, faceGrid, label


## mean images for standardization
def get_mean_image(data_dir, file_name):
    image_mean = np.array(sio.loadmat(data_dir + file_name)['image_mean'])
    image_mean = image_mean.reshape(3, 224, 224)
    return image_mean.mean(1).mean(1)

## define grid
def makeGrid(gridSize, params):
    gridLen = gridSize[0] * gridSize[1]
    grid = np.zeros([gridLen, ], np.float32)

    indsY = np.array([i // gridSize[0] for i in range(gridLen)])
    indsX = np.array([i % gridSize[0] for i in range(gridLen)])
    condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
    condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
    cond = np.logical_and(condX, condY)

    grid[cond] = 1
    return grid

## define transform
class ToPILImage(object):
    def __call__(self, input):
        input = transforms.ToPILImage()(input)
        return input


class ToNumpy(object):
    def __call__(self, input):
        return np.array(input)


class SubtractMean(object):
    def __init__(self, meanImg):
        self.meanImg = meanImg

    def __call__(self, input):
        input = input - self.meanImg
        return input


class ToTensor(object):
    def __call__(self, input):
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        input = np.moveaxis(input[:, :, :], -1, 0)
        input = torch.from_numpy(input / 255)
        return input

