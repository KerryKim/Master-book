"""
-------------------------------------
denoising
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task denoising \
        --opts random 30.0
python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task denoising \
        --opts random 30.0
-------------------------------------
inpainting
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task inpainting \
        --opts random 0.5
python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task inpainting \
        --opts random 0.5
-------------------------------------
super_resolution
-------------------------------------
python  train.py \
        --mode train \
        --network unet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0
python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0
python  train.py \
        --mode train \
        --network srresnet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0 0.0
"""

## 라이브러리 추가하기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms

def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir_train, 'png'))
        # os.makedirs(os.path.join(result_dir_train, 'numpy'))


    ## 네트워크 학습하기
    if mode == 'train':
        # transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), RandomFlip()])
        # transform_val = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)])

        # Label image를 Tanh()의 dynamic range와 동일하게 만들기 위해 Normalization transform 추가
        transform_train = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_train = Dataset(data_dir=data_dir, transform=transform_train, task=task, opts=opts)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)


    ## 네트워크 생성하기
    if network == "DCGAN":  # 2개의 network 필요
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        # weight 초기화
        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    # Generative Adversarial Networks 논문 검색후 loss fn 구현
    # Generator 모델의 경우에는 이 지폐가 진짜인지 가짜인지 구분해내는 Binary cross entropy loss를 사용

    fn_loss = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    # 네트워크가 2개 이므로 2개의 네트워크 사용

    '''
    논문에서는 모멘텀 값을 수정하여 사용
    
    we found leaving the momentum term beta1 at the suggested value of 0.9 resulted in training oscillation
    and instability whie reducing it to 0.5 helped stabilzie training.
    '''
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.ADam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG,
                                                        netD=netD, optimG=optimG, optimD=optimD)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG.train()
            netD.train()

            '''
            In other words, D and G play the following two-player minimax game with value function V(G,D):
            Loss funtion:
            min max V (D,G) = E [log D(x)] + E [log(1-D(G(z))]
    
            Discriminator에 들어가는 term이 2개가 있는데 하나는 x고 하나는 Generator로 형성된 fake 이미지(즉, G(z))이다.
            '''

            loss_G_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            for batch, data in enumerate(loader_train, 1):
                # forward pass
                label = data['label'].to(device)
                # input = data['input'].to(device)
                input = torch.randn(label.shape[0], 100, 1, 1, ).to(device)

                output = netG(input)

                # backward netD
                # 처음에는 Discriminator에 대한 back propagation 수행
                # 이렇게 하면 Discriminator의 모든 파라미터들의 require_grad term이 True로 설정되어 업데이트가 실행된다.
                set_requires_grad(netD, True)
                optimD.zero_grad()

                # Discriminator에 real과 generator output을 입력시켜 그 결과가 true인지 false인지 확인하기
                pred_real = netD(label)  # 우리는 label을 입력했을때 True라는 값을 얻기를 원한다.
                pred_fake = netD(output.detach())  # generator output을 discriminator에 입력했을 때 False를 얻길 원한다.

                '''
                여기서 중요한것은 discriminator의 backward 파트이기 때문에
                Generator output에 detach라는 루틴을 적용시켜서 discriminator로부터 propagation되는
                backward루틴이 generator까지 넘어가지 않도록 끊어주는 역할을 한다.
    
                output에 detach 루틴을 걸어줘야 온전히 discriminator에 대해서만 업데이트가 진행된다.
                '''

                # prediction된 real을 0, prediction을 fake를 1로 바꾸어 입력한다.
                loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))
                loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))

                # loss를 2개로 설정했기 때문에 2개의 평균값을 최종로스로 (임의) 설정한다.
                loss_D = 0.5 * (loss_D_real + loss_D_fake)
                loss_D.backward()

                optimD.step()

                # backward netG
                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(output)

                '''
                이번에는 Discriminator와 다르게 Generator에 의해 생성된 이미지가 True와 같이 보이길 원하기 때문에
                output을 0(real)로 바꾸어 입력한다.
                '''
                loss_G = fn_loss(pred_fake, torch.ones_like(pred_fake))

                loss_G.backward()
                optimG.step()

                # 손실함수 계산
                loss_G_train += [loss_G.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN %.4f | DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

                if batch % 20 == 0:
                    # Tensorboard 저장하기
                    # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                    # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                    # output의 경우 DCGAN 모델에서 tanh를 이용해서 -1~1로 normalize 했기 때문에
                    # fn_denorm을 이용해서 0~1사이로 range를 transfrom시킨다.
                    output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
                    output = np.clip(output, a_min=0, a_mmax=1)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(),
                               cmap=cmap)
                    writer_train.add_image('output', output, id, dataformats='NHWC')

            # tensorboard에 loss를 저장하는 부분
            writer_train.add_scalar('loss', np.mean(loss_G_train), epoch)
            writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
            writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

            if epoch % 50 == 0:
                save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close()


def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir_test, 'png'))
        os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == 'test':
        # transform_test = transforms.Compose([Normalization(mean=0.5, std=0.5)])

        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

        dataset_test = Dataset(data_dir=data_dir, transform=transform_test, task=task, opts=opts)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)

        num_batch_test = np.ceil(num_data_test / batch_size)

    ## 네트워크 생성하기
    if network == "DCGAN":  # 2개의 network 필요
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

        # weight 초기화
        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## 손실함수 정의하기
    # Generative Adversarial Networks 논문 검색후 loss fn 구현
    # Generator 모델의 경우에는 이 지폐가 진짜인지 가짜인지 구분해내는 Binary cross entropy loss를 사용

    fn_loss = nn.BCELoss().to(device)

    ## Optimizer 설정하기
    # 네트워크가 2개 이므로 2개의 네트워크 사용

    '''
    논문에서는 모멘텀 값을 수정하여 사용

    we found leaving the momentum term beta1 at the suggested value of 0.9 resulted in training oscillation
    and instability whie reducing it to 0.5 helped stabilzie training.
    '''
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.ADam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if mode == 'test':

        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG,
                                                    optimD=optimD)

        with torch.no_grad():
            netG.eval()

            input = torch.randn(batch_size, 100, 1, 1).to(device)
            output = netG(input)

            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            for j in range(output.shape[0]):
                id = j

                output_ = output[j]
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                output_ = np.clip(output_, a_min=0, a_max=1)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)
