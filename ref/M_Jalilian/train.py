import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from NetModel import *

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.criterion import CriterionPixelWise, CriterionPairWiseforWholeFeatAfterPool
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Resize

dir_img = 'D:/Dataset/imgs/'
dir_mask = 'D:/Dataset/masks/'
dir_checkpoint = 'checkpoints/'

"""
依靠训练集数据来训练网络
注意，由于用于输入训练的图像大小不一，batch size请保持为1
"""
def train_net(net,
              device,
              epochs=5,
              batch_size=1,                             # 注意，由于用于输入训练的图像大小不一，batch size请保持为1
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale, force_samesize=True)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    # 3. Create data loaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)              # set pin_memory as False to avoid unnormal CPU usage increase
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=True) # set pin_memory as False to avoid unnormal CPU usage increase

    # 4. Initialize logging
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # 5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    # 6. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                _torchResize = Resize([1280,1280])
                masks_pred = _torchResize(masks_pred)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    """
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    """
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


"""
使用知识蒸馏来训练学生网络
注意，由于用于输入训练的图像大小不一，batch size请保持为1
"""
def train_net_withKD(student,
                     teacher,
                     device,
                     epochs=5,
                     batch_size=1,
                     lr=0.001,
                     val_percent=0.1,
                     save_cp=True,
                     img_scale=1):
    # Check the output classes of student and teacher network
    assert (student.n_classes == 1) & (teacher.n_classes == 1),\
        'The output classes of student and teacher must be the 1'
    assert (student.n_channels == teacher.n_channels),\
        'The input channels of student and teacher must be the same'

    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    # 3. Create data loaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)              # set pin_memory as False to avoid unnormal CPU usage increase
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False, drop_last=True) # set pin_memory as False to avoid unnormal CPU usage increase

    # 4. Initialize logging
    writer = SummaryWriter(comment=f'KD_LR_{lr}_BS_{batch_size}_SCALE_{img_scale}_LT_{args.lambda_t}_LPI_{args.lambda_pi}_LPA_{args.lambda_pa}')
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Lambda teacher:  {args.lambda_t}
        Lambda PixelWise:{args.lambda_pi}
        Lambda PairWise: {args.lambda_pa}
    ''')

    # 5. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(student.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if student.n_classes > 1 else 'max', patience=2)
    criterion_groundtrue = nn.BCEWithLogitsLoss()
    criterion_teacher_pixelwise = nn.BCEWithLogitsLoss()    # CriterionPixelWise()
    criterion_teacher_pairwise = CriterionPairWiseforWholeFeatAfterPool()
    global_step = 0

    # 6. Begin training
    for epoch in range(epochs):
        student.train()
        teacher.eval()

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert (imgs.shape[1] == student.n_channels), \
                    f'Studet network has been defined with {student.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if student.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # 获取teacher和student的输出，注意此处要将teacher的预测结果进行预处理
                with torch.no_grad():
                    teacher_pred = torch.sigmoid(teacher(imgs))
                    # teacher_pred = (teacher_pred > 0.5).float()
                student_pred = student(imgs)

                # 计算误差
                loss_groundtrue = criterion_groundtrue(student_pred, true_masks)
                loss_pi = criterion_teacher_pixelwise(student_pred, teacher_pred)
                loss_pa = criterion_teacher_pairwise(student_pred, teacher_pred)
                loss_teacher = (args.lambda_pi * loss_pi) + (args.lambda_pa * loss_pa)
                global_loss = ((1 - args.lambda_t) * loss_groundtrue) + (args.lambda_t * loss_teacher)

                # test
                logging.info(f'\nloss_pi:{loss_pi},loss_pa:{loss_pa},loss_teacher:{loss_teacher}\n'
                             f'loss_gt:{loss_groundtrue},loss_global:{global_loss}')

                # 将loss写入log
                writer.add_scalar('Loss/with teacher', loss_teacher.item(), global_step)
                writer.add_scalar('Loss/with groundtrue', loss_groundtrue.item(), global_step)
                writer.add_scalar('Loss/global', global_loss.item(), global_step)
                writer.add_scalar('Loss/loss_pi', loss_pi.item(), global_step)
                writer.add_scalar('Loss/loss_pa', loss_pa.item(), global_step)

                # 设置进度条
                pbar.set_postfix(**{'Global loss': global_loss.item(),
                                    'Teacher loss': loss_teacher.item(),
                                    'GroundTrue loss': loss_groundtrue.item()})

                # 优化
                optimizer.zero_grad()
                global_loss.backward()
                nn.utils.clip_grad_value_(student.parameters(), 0.1)
                optimizer.step()

                # 更新进度条，因为可能batch不等于1，所以依照tensor的N来更新数量
                pbar.update(imgs.shape[0])

                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in student.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(student, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    # 记录验证结果
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/student_pred', torch.sigmoid(student_pred) > 0.5, global_step)
                    writer.add_images('masks/teacher_pred', torch.sigmoid(teacher_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(student.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', dest='model', required=True,
                        help="Specify the model to be used")
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    parser.add_argument('--use-KD', dest='KD', action='store_true',
                        help='Declare whether knowledge distillation is enabled',
                        default=False)
    parser.add_argument('-t', '--tmodel', dest='tmodel', type=str,
                        help='Specify the teacher model to be used, only enabled when use knowledge distillation')
    parser.add_argument('--lambda-t', dest='lambda_t', type=float, default=0.9,
                        help='The percentage of teacher loss(soft loss) in the final loss function (0 ~ 1)')
    parser.add_argument('--lambda-pi', dest='lambda_pi', type=float, default=10.0,
                        help='The ratio of PixelWise before adding it to the soft loss')
    parser.add_argument('--lambda-pa', dest='lambda_pa', type=float, default=0.5,
                        help='The ratio of PairWise before adding it to the soft loss')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    # 选择要使用的网络
    net = torch.nn.Module()
    if args.model == 'RefineNet':
        net = RefineNet(Bottleneck, [3, 4, 23, 3], num_classes=1, n_channels=3, n_classes=1)
    else:
        logging.error(f'Undefined model {args.model}, trainning terminated')
        sys.exit(0)

    # 可选的加载模型权重
    if args.load:
        net.load_state_dict(
            torch.load(args.load + '.pth', map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    # 移动训练模型到设备
    net.to(device=device)

    # 可选知识蒸馏
    if not args.KD:
        '''
        独立训练
        '''
        logging.info(f'trainning {args.model}:\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)\n'
                     f'Bilinear upscaling')

        # 开始独立训练
        try:
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      img_scale=args.scale,
                      val_percent=args.val / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    else:
        '''
        使用知识蒸馏训练模型
        '''
        # 选择teacher
        teacher_net = torch.nn.Module()
        if args.tmodel == 'VSN':
            teacher_net = VSN(n_channels=3, n_classes=1)
        elif args.tmodel == 'VSNLite4M':
            teacher_net = VSNLite4M(n_channels=3, n_classes=1)
        else:
            logging.error(f'Undefined teacher {args.tmodel}, trainning terminated')
            sys.exit(0)

        # 加载teacher权重
        teacher_net.load_state_dict(torch.load('TMODEL_' + args.tmodel + '.pth', map_location=device))
        logging.info(f'teacher model loaded from TMODEL_{args.tmodel}.pth')

        # move teacher to device
        teacher_net.to(device=device)

        logging.info(f'Using KD trainning student {args.model}:\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)')
        logging.info(f'Using {args.tmodel} as teacher:\n'
                     f'\t{teacher_net.n_channels} input channels\n'
                     f'\t{teacher_net.n_classes} output channels (classes)')

        # start KD trainning
        try:
            train_net_withKD(student=net,
                             teacher=teacher_net,
                             epochs=args.epochs,
                             batch_size=args.batchsize,
                             lr=args.lr,
                             device=device,
                             img_scale=args.scale,
                             val_percent=args.val / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
