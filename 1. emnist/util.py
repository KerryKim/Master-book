import os
import numpy as np
import pandas as pd

import torch

from datetime import datetime

## from tensor to numpy function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()

## 리스트 펼치기 함수
def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    return result

## 확률 결과값을 숫자로 변환하
def prob_to_digit(output, batch_size):
    np_output = fn_tonumpy(output)
    lst_output = []

    for i in range(batch_size):
        nb_output = np.argmax(np_output[i, :])
        lst_output.append(nb_output)

    return lst_output

## 정확도 계산하기
# this output is digit (numpy_output, number_output)
def cal_accuracy(lst_output, data):
    lst_output = fn_tonumpy(torch.FloatTensor(lst_output))
    label = fn_tonumpy(data['label'].to('cpu'))

    acc = np.sum(lst_output == label) / len(label)

    return acc

## 제출파일 저장하기
def save_submission(result_dir, prediction, epoch, batch):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    suffix = 'epoch{}_batch{}_date{}'.format(epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    pred = flatten(prediction)
    submission = pd.read_csv('./datasets/submission.csv')
    submission['digit'] = fn_tonumpy(torch.LongTensor(pred))
    submission.to_csv('./result/submission_{}.csv'.format(suffix), index=False)

## 네트워크 저장하기
def save_model(ckpt_dir, net, optim, epoch, batch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    suffix = 'epoch{}_batch{}_date{}'.format(epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "{}/model_{}.pth".format(ckpt_dir, suffix))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('_batch')[0])

    return net, optim, epoch