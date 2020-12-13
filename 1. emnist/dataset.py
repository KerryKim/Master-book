## import libraries
import numpy as np
import torch

from torchvision import transforms

## define dataset

'''
This dataset example is used of .csv data
Label(digit) is a 2nd colum
Input(pixel values) is 3: colums
Image size is 28*28*1 pixels
'''

class TrainDataset(torch.utils.data.Dataset):
    # dataset preprocessing
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

        # lst_label, lst_input for __getitem__()
        label = self.data['digit']
        lst_label = list(label.values)

        lst_input = []
        df_input = self.data.iloc[:, 3:]
        for i in range(len(df_input)):
            temp = df_input.iloc[i, :].values.reshape(28, 28)
            temp = np.where(temp >= 4, temp, 0) #data preprocessing
            lst_input.append(temp)

        self.lst_label = lst_label
        self.lst_input = lst_input

    # dataset length
    def __len__(self):
        return len(self.data)

    # bring one sample from Dataset
    def __getitem__(self, index):
        # input data return by index
        label = self.lst_label[index]
        input = self.lst_input[index]

        label = np.array(label)
        input = (np.array(input)).astype(np.float32)

        input = input / 255.0

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

        lst_label = []  # To pass transform-error
        lst_input = []

        df_input = self.data.iloc[:, 2:]
        for i in range(len(df_input)):
            temp = df_input.iloc[i, :].values.reshape(28, 28)
            temp = np.where(temp >= 4, temp, 0)
            lst_input.append(temp)

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.lst_label
        input = self.lst_input[index]

        label = np.array(label)
        input = (np.array(input)).astype(np.float32)
        input = input / 255.0

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label' : label}

        if self.transform:
            data = self.transform(data)

        return data

## define transform

'''
Modeldml nn.Conv2D는 (샘플수x채널수x세로x가로)의 4차원 Tensor를 입력을 기본으로 하고
샘플 수에 대한 차원이 없을땐 (채널수x세로x가로)만 입력해도 되는 듯하다.

torchvision에서 제공되는 transforms 함수를 쓰기 위해서는 PILImage 변환이 필요하다.
'''

class ToPILImage(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        ToPIL = transforms.ToPILImage()
        data = {'label': label, 'input': ToPIL(input)}
        return data


class RandomRotation(object):
    def __init__(self, degree=5):
        self.degree = degree

    def __call__(self, data):
        label, input = data['label'], data['input']
        rotate = transforms.RandomRotation(self.degree)
        data = {'label': label, 'input': rotate(input)}
        return data


class RandomAffine(object):
    # if degree=0, Affine is shift
    def __init__(self, degree=0, translate=(0.1, 0.1)):
        self.degree = degree
        self.translate = translate

    def __call__(self, data):
        label, input = data['label'], data['input']
        affine = transforms.RandomAffine(self.degree, self.translate)
        data = {'label': label, 'input': affine(input)}
        return data


class ToNumpy(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        data = {'label': label, 'input': np.array(input)}
        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        data = {'label': label, 'input': input}
        return data


class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # When transforms from PIL to Numpy, an axis was gone.
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # (width, height, channel) → (channel, width, height)
        # The order is changed, not poistion
        input = np.moveaxis(input[:, :, :], -1, 0)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
        return data