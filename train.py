from __future__ import division

import os
import skimage
import cv2
import torch
import re
import time
# import tta_inference

from thop import profile
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import tqdm
from torchstat import stat
import tools.transform as tr
from tools.dataloader import IsprsSegmentation
from torchvision.utils import make_grid
import tools
from tensorboardX import SummaryWriter
from inference import slide_pred
from networks.get_net import get_net
from config import *
from tools.cal_iou import evaluate
from tools.losses import get_loss
from tools.draw_pic import draw_pic
import numpy as np
from tools.utils import label_mapping, accuracy
import torchsummary
from torchstat import stat

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_list = [i for i in range(len(gpu_id.split(',')))]

def parse_color_table(color_txt):
    f = open(color_txt, 'r').readlines()
    color_table = []
    for info in f:
        x = info.split('#')[0].split('/')
        color_table.append((int(x[0]), int(x[1]), int(x[2])))
    return color_table


def main():
    composed_transforms_train = standard_transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.5)),
        # tr.RandomResizedCrop(img_size),
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])  # data pocessing and data augumentation
    composed_transforms_val = standard_transforms.Compose([
        tr.FixedResize(img_size),
        tr.Normalize(mean=mean, std=std),
        tr.ToTensor()])  # data pocessing and data augumentation

    road_train = IsprsSegmentation(base_dir=root_data, split='train', transform=composed_transforms_train)  # get data
    trainloader = DataLoader(road_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)  # define traindata

    road_val = IsprsSegmentation(base_dir=root_data, split='test', transform=composed_transforms_val)  # get data
    valloader = DataLoader(road_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)  # define traindata

    if use_gpu:
        model = torch.nn.DataParallel(frame_work, device_ids=gpu_list)  # use gpu to train
        model_id = 0
        if find_new_file(model_dir) is not None:
            model.load_state_dict(torch.load(find_new_file(model_dir)))
            print('load the model %s' % find_new_file(model_dir))
            model_id = re.findall(r'\d+', find_new_file(model_dir))
            model_id = int(model_id[0])

        model = model.cuda()

    else:
        model = frame_work
        model_id = 0
        if find_new_file(model_dir) is not None:
            model.load_state_dict(torch.load(find_new_file(model_dir)))
            print('load the model %s' % find_new_file(model_dir))
            model_id = re.findall(r'\d+', find_new_file(model_dir))
            model_id = int(model_id[0])

    criterion = get_loss(loss_type)  # define loss
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)  # define optimizer
    writer = SummaryWriter(os.path.join(save_dir_model, 'runs'))

    f = open(os.path.join(save_dir_model, 'log.txt'), 'w')
    for epoch in range(epoches):
        model.train()
        running_loss = 0.0
        lr = adjust_learning_rate(base_lr, optimizer, epoch, model_id, power)  # adjust learning rate
        batch_num = 0
        for i, data in tqdm.tqdm(enumerate(trainloader)):  # get data
            images, labels = data['image'], data['gt']
            i += images.size()[0]


            labels = labels.view(images.size()[0], img_size, img_size).long()
            if use_gpu:
                """
                images = images.to(device)
                nsdms = nsdms.to(device)
                labels = labels.to(device)
                """
                images = images.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
            if model_name != 'pspnet':
                outputs = model(images)  # get prediction
            else:
                outputs, _ = model(images)

            """ print("===================")
                flops, params = profile(model, inputs=(images, nsdms))
                print('flops: ', flops, 'params: ', params)
                print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
                print("===================")
            """
            losses = criterion(outputs, labels)  # calculate loss
            losses.backward()
            optimizer.step()
            running_loss += losses

            if i % 50 == 0:
                grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image)
                grid_image = make_grid(
                    tools.utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()),
                    3,
                    normalize=False,
                    range=(0, 255))
                writer.add_image('predicted label', grid_image)
                grid_image = make_grid(
                    tools.utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()),
                    3,
                    normalize=False, range=(0, 255))
                writer.add_image('Ground truth label', grid_image)
            batch_num += images.size()[0]

        print('epoch is {}, train loss is {}'.format(epoch, running_loss.item() / batch_num))
        writer.add_scalar('learning_rate', lr, epoch)
        writer.add_scalar('train_loss', running_loss / batch_num, epoch)

        if epoch % save_iter == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, '%d.pth' % (model_id + epoch + 1)))
            # val_acc = eval(model, criterion, epoch)
            # writer.add_scalar('val_acc', val_acc, epoch)
            # cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}, val_acc:{}\n'.format(str(epoch), str(lr), str(running_loss.item() / batch_num),  str(val_acc))
            cur_log = 'epoch:{}, learning_rate:{}, train_loss:{}\n'.format(str(epoch), str(lr), str(running_loss.item() / batch_num))
            print(cur_log)
            f.writelines(str(cur_log))

def eval(model, criterion, epoch):

    model.eval()
    if val_visual:
        if os.path.exists(os.path.join(save_dir, 'val_visual')) is False:
            os.mkdir(os.path.join(save_dir, 'val_visual'))
        if os.path.exists(os.path.join(save_dir, 'val_visual', str(epoch))) is False:
            os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch)))
        if os.path.exists(os.path.join(save_dir, 'val_visual', str(epoch), 'color_big')) is False:
            os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch), 'color_big'))
        if os.path.exists(os.path.join(save_dir, 'val_visual', str(epoch), 'gray_big')) is False:
            os.mkdir(os.path.join(save_dir, 'val_visual', str(epoch), 'gray_big'))
    with torch.no_grad():
        batch_num = 0
        test_path = '../Vaihingen/img_test'
        test_imgs = os.listdir(test_path)
        color_txt = '../Vaihingen/color_table_isprs.txt'
        color_table = parse_color_table(color_txt)
        test_gt = '../Vaihingen/gt_nobd_2'

        for name in tqdm.tqdm(test_imgs):
            output = slide_pred(model=model, image_path=os.path.join(test_path, name),num_classes=6,crop_size=512, overlap=256, scales=[1.0], flip=True)
            pred_gray = torch.argmax(output, 1)
            pred_gray = pred_gray[0].cpu().data.numpy().astype(np.int32)
            pred_vis = label_mapping(pred_gray)
            skimage.io.imsave(os.path.join(save_dir, 'val_visual', str(epoch), 'gray_big', name), pred_gray + 1, color_table=color_table)
            sample_path=os.path.join(save_dir, 'val_visual', str(epoch), 'color_big', name)
            cv2.imwrite(sample_path, pred_vis)

        val_acc  = evaluate(test_gt, os.path.join(save_dir, 'val_visual', str(epoch), 'gray_big'), num_class)

    return  val_acc

def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None

def adjust_learning_rate(base_lr, optimizer, epoch, model_id, power):
    lr = base_lr * (power ** ((epoch + model_id) // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    frame_work = get_net(model_name, 3, num_class, img_size)
    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    main()
