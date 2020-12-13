import os, shutil

import torch

from datetime import datetime

## save model
def save_model(ckpt_dir, net, optim, num_epoch, epoch, batch, current_best, is_best):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    suffix = 'model_epoch{}_batch{}_date{}'.format(epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    filename = os.path.join(ckpt_dir, 'checkpoint')
    best_filename = os.path.join(ckpt_dir, 'best_checkpoint')
    final_filename = os.path.join(ckpt_dir, 'final_' + suffix)

    # save model every epoch
    torch.save({'epoch' : epoch, 'net': net.state_dict(), 'optim': optim.state_dict(),
                'current_best' : current_best}, filename)

    # leave only best model
    if is_best:
        shutil.copyfile(filename, best_filename)

    if num_epoch == epoch:
        shutil.copyfile(best_filename, final_filename)

## load model
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

## learning rate decay
def adjust_learning_rate(lr, optim, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optim.state_dict()['param_groups']:
        param_group['lr'] = lr