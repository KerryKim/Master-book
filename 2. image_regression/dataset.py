import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from util import *

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)

        # 확장자가 jpg, png파일만 리스트에 담기, |는 and인듯
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png')]

        lst_data.sort()

        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        # 이미지가 numpy array가 아니기 때문에 png파일이기 때문에 matplotlib를 이용
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        # input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape

        # 저장된 이미지가 세로 혹은 가로로 저장되어 있어 이것을 항상 가로로 긴 형태로 정리
        if sz[0] > sz[1]:
            img = img.transpose((1, 0, 2))

        # 사실 255로 normalize하는 것은 unit 8인 경우만 가능
        if img.dtype == np.unit8:
            img = img/255.0

        if img.ndim == 2:
            label = img[:, :, np.newaxis]

        label = img

        data = {'label': label}

        if self.task == "denoising":
            data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])
        elif self.task == "inpainting":
            data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])

        if self.transform:
            data = self.transform(data)

        # super_resolution의 경우 down_sampling으로 인해 input/output size가 다를 수 있다.
        if self.task == "super_resolution":
            data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])

        data = self.to_tensor(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)
        #
        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        for key, value in data.imtes():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # label, input = data['label'], data['input']
        #
        # label = (label - self.mean) / self.std
        # input = (input - self.mean) / self.std
        #
        # data = {'label': label, 'input': input}

        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            # label = np.fliplr(label)
            # input = np.fliplr(input)

            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            # label = np.flipud(label)
            # input = np.flipud(input)

            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        # data = {'label': label, 'input': input}

        return data

class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        # input, label = data['input'], data['label']
        #
        # h, w = input.shape[:2]

        h, w = data['label'].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        # input = input[id_y, id_x]
        # label = label[id_y, id_x]
        #
        # data = {'input' : input, 'label' : label}

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data
