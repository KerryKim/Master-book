## 라이브러리 추가하기
import os
import argparse
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# self-class
from model import *
from dataset import *
from util import *

## 랜덤시드 고정하기
# seed 값을 고정해야 hyper parameter 바꿀 때마다 결과를 비교할 수 있습니다.
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 0
seed_everything(seed)

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the Net",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=40, type=int, dest="num_epoch")

parser.add_argument("--nch", default=3, type=int, dest="nch")   # RGB channels
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--train/test_mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args, unknown = parser.parse_known_args()

## 트레이닝 파라메터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

nch = args.nch
nker = args.nker

mode = args.mode
train_continue = args.train_continue

learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("==========DEFAULT=========")
print("learning type: %s" % learning_type)

print("train/test_mode: %s" % mode)
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

# print("task: %s" % task)
# print("opts: %s" % opts)

# print("network: %s" % network)
print("learning type: %s" % learning_type)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("device: %s" % device)

## 네트워크 학습하기

'''
    sklearn의 train_test_split 사용시 output 순서
    4분류 : train_input, valid_input, train_label,valid_label
    2분류 : train, valid
'''

if mode == 'train':
    transform_train = transforms.Compose([ToPILImage(), RandomRotation(degree=5), RandomAffine(degree=0, translate=(0.1, 0.1)),
                                          ToNumpy(), Normalization(mean=0.5, std=0.5), ToTensor()])
    transform_val = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    load_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train, val = train_test_split(load_data, test_size=0.125, random_state=seed)

    dataset_train = TrainDataset(train, transform=transform_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = TrainDataset(val, transform=transform_val)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=Trye, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)  # np.ceil은 올림 함수. Ex> 4.2 → 5 로 변환
    num_batch_val = np.ceil(num_data_val / batch_size)

else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    load_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    dataset_test = TestDataset(load_data, transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = SimpleNet().to(device)

## 손실함수 정의하기
fn_loss = nn.CrossEntropyLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
# When tensor changes to numpy, need to move cpu.
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

print("==========MODEL ARCHITECTURE=========")
print(net)
## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    print("==========TRAINING MODE=========")

    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    # list loss & accuracy for print
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []   # this is 1 batch loss array

        # calculate sum of 1 epoch for list loss & accuracy
        loss_epoch_train = 0
        acc_epoch_train = 0

        # batch is counting index. data is loader_train value.
        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]
            loss_batch_train = np.mean(loss_arr)
            loss_epoch_train += loss_batch_train

            # 정확도 계산
            lst_output_train = prob_to_digit(output, batch_size)
            acc_batch_train = cal_accuracy(lst_output_train, data)
            acc_epoch_train += acc_batch_train

        train_loss.append(loss_epoch_train / num_batch_train)
        train_acc.append(acc_epoch_train / num_batch_train)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            loss_epoch_val = 0
            acc_epoch_val = 0

            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)

                loss_arr += [loss.item()]
                loss_batch_val = np.mean(loss_arr)
                loss_epoch_val += loss_batch_val

                lst_output_val = prob_to_digit(output, batch_size)
                acc_batch_val = cal_accuracy(lst_output_val, data)
                acc_epoch_val += acc_batch_val

            val_loss.append(loss_batch_val / num_batch_val)
            val_acc.append(acc_batch_val / num_batch_val)

        # print result for one epoch
        print("EPOCH: {}/{} | ".format(epoch, num_epoch), "TRAIN_LOSS: {:4f} | ".format(train_loss[-1]),
              "TRAIN_ACC: {:4f} | ".format(train_acc[-1]), "VAL_LOSS: {:4f} | ".format(val_loss[-1]),
              "VAL_ACC: {:4f}".format(val_acc[-1]))

        if epoch % num_epoch == 0:
            save_model(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=num_epoch, batch=batch_size)

# TEST MODE
else:
    print("==========TEST MODE=========")

    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        pred = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            input = data['input'].to(device)

            output = net(input)

            lst_output = prob_to_digit(output, batch_size)
            pred.append(lst_output)

            print("TEST: BATCH %04d / %04d" %
                  (batch, num_batch_test))

        # submission
        if batch % num_batch_test == 0:
            save_submission(result_dir=result_dir, prediction=pred, epoch=num_epoch, batch=batch_size)

    print("AVERAGE TEST: BATCH %04d / %04d" %
          (batch, num_batch_test))