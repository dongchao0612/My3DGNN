import cv2
import os
import sys
import time
import numpy as np
import datetime
import logging
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model
import config
from tqdm import tqdm
import argparse
from PIL import Image

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('3dgnn')
    # 指定循环次数
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of epoch')
    # 指定批量大小
    parser.add_argument('--batchsize', type=int, default=2, help='batch size in training')
    # 修改default,可以使用训练好的参数
    parser.add_argument('--pretrain', type=str, default="./train/3dgnn_finish.pth",
                        help='Direction for pretrained weight')
    # 指定GPU,一般从0开始编号
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    return parser.parse_args()


def main(args):
    '''指定GPU'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''指定日志文件的存放位置'''
    logger = logging.getLogger('3dgnn')
    log_path = './eval/' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H')).replace(' ', '/') + '/'
    print('日志路径:', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    hdlr = logging.FileHandler(log_path + 'log.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("加载数据...")
    print("加载数据...")

    '''创建文件夹，保存前向传播后的预测结果'''
    eval_path = './eval/' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H')).replace(' ', '/') + '/'
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    '''dataset_va是预测集'''
    dataset_va = Dataset(flip_prob=config.flip_prob, crop_type='Random', crop_size=config.crop_size)
    dataloader_va = DataLoader(dataset_va, batch_size=args.batchsize, shuffle=False, num_workers=config.workers_va,
                               drop_last=True, pin_memory=True)
    cv2.setNumThreads(config.workers_tr)

    '''日志文件需要添加的信息'''
    logger.info("准备模型...")
    print("准备模型...")

    '''模型初始化'''
    model = Model(config.nclasses, config.mlp_num_layers, config.use_gpu)

    '''dim表示维度，dim=0,表示行，dim=1，表示列'''
    softmax = nn.Softmax(dim=1)

    '''使用cuda加速'''
    if config.use_gpu:
        model = model.cuda()
    '''评估/预测,对输入数据进行评估/预测'''
    model_to_load = args.pretrain
    '''判断使用原来训练过的模型参数，还是从零开始训练'''
    if model_to_load:
        logger.info("加载旧模型...")
        print("加载旧模型...")
        model.load_state_dict(torch.load(model_to_load))
    else:
        logger.info("从头开始训练...")
        print("从头开始训练...")

    '''预测集（也是训练集），重新放入网络进行前向传播，评估损失和交并比'''

    model.eval()

    '''torch.no_grad()函数使得程序不计算梯度，只进行前向传播，用在预测中正合适'''
    with torch.no_grad():
        index = 1
        '''batch_idx指目前是第几个batch'''
        for batch_idx, rgbd_label_xy in enumerate(dataloader_va):
            # torch.Tensor 类型 [6, 640, 480, 6] 对应数据集中的 rgb_hha 矩阵
            # 第一个6是batch大小，640和480是宽和高，第二个6是因为其为rgb与hha的拼接
            x = rgbd_label_xy[0]
            # torch.Tensor 类型 [6, 640, 480, 2] 对应数据集中的 xy 矩阵
            # 第一个6是batch大小，640和480是宽和高，2是因为 xy 数据集只有两个维度，且为全零矩阵
            xy = rgbd_label_xy[2]
            # torch.Tensor 类型 [6, 640, 480] 第一个6是batch大小，640和480是宽和高
            target = rgbd_label_xy[1].long()
            x = x.float()
            xy = xy.float()
            '''permute函数用于转换Tensor的维度，contiguous()使得内存是连续的'''
            input = x.permute(0, 3, 1, 2).contiguous()
            xy = xy.permute(0, 3, 1, 2).contiguous()
            if config.use_gpu:
                input = input.cuda()
                xy = xy.cuda()
            '''经过网络，计算输出, 维度为 ([6, 14, 640, 480])'''
            output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy, use_gnn=config.use_gnn)

            '''pred维度为 ([6, 640, 480, 14]), 连续内存'''
            pred = output.permute(0, 2, 3, 1).contiguous()
            '''源程序没有，专门存放这个batch的大小，后面循环用到'''
            name_for_txt = len(pred)  # batchsize
            '''此时pred维度为 ([1843200, 14]), 其中1843200=6*640*480  config.nclasses=14'''
            pred = pred.view(-1, config.nclasses)
            '''每一行进行softmax运算，相当于对每一个像素的分类进行softmax运算'''
            pred = softmax(pred)
            '''pred_max_val, pred_arg_max都是1843200维，分别存储每个像素最大的分类概率及分类'''
            pred_max_val, pred_arg_max = pred.max(1)
            '''将分类数据保存为数组，并且改变形状，使得每行代表一张图片'''
            result = pred_arg_max.cpu().numpy().reshape(name_for_txt, 307200)
            Width = 640
            Height = 480
            '''创建空矩阵，用于存放一张图片每个像素的分类数据'''
            Empty_array = np.zeros((Width, Height, 3), dtype=np.uint8)

            for ii in range(name_for_txt):
                row_Frame = result[ii]  # 将一张图片的数据单独保存 取出来一列像素值
                for w in range(Width):
                    for h in range(Height):
                        '''判断属于哪一类，不一样的类赋予不同的颜色'''
                        if row_Frame[w * Height + h] == 0:
                            # 未知类，RGB为0，黑色
                            Empty_array[w, h, 0] = 0
                            Empty_array[w, h, 1] = 0
                            Empty_array[w, h, 2] = 0
                        elif row_Frame[w * Height + h] == 1:
                            # beam类，RGB为 石板灰
                            Empty_array[w, h, 0] = 112
                            Empty_array[w, h, 1] = 128
                            Empty_array[w, h, 2] = 105
                        elif row_Frame[w * Height + h] == 2:
                            # board类，RGB为 马棕色
                            Empty_array[w, h, 0] = 139
                            Empty_array[w, h, 1] = 69
                            Empty_array[w, h, 2] = 19
                        elif row_Frame[w * Height + h] == 3:
                            # bookcase类，RGB为 乌贼墨棕色
                            Empty_array[w, h, 0] = 94
                            Empty_array[w, h, 1] = 38
                            Empty_array[w, h, 2] = 18
                        elif row_Frame[w * Height + h] == 4:
                            # ceiling类，RGB为
                            Empty_array[w, h, 0] = 220
                            Empty_array[w, h, 1] = 220
                            Empty_array[w, h, 2] = 220
                        elif row_Frame[w * Height + h] == 5:
                            # chair类，RGB为 玫瑰红
                            Empty_array[w, h, 0] = 188
                            Empty_array[w, h, 1] = 143
                            Empty_array[w, h, 2] = 143
                        elif row_Frame[w * Height + h] == 6:
                            # clutter类，RGB为 镉红
                            Empty_array[w, h, 0] = 227
                            Empty_array[w, h, 1] = 23
                            Empty_array[w, h, 2] = 13
                        elif row_Frame[w * Height + h] == 7:
                            # column类，RGB为 紫色
                            Empty_array[w, h, 0] = 160
                            Empty_array[w, h, 1] = 32
                            Empty_array[w, h, 2] = 240
                        elif row_Frame[w * Height + h] == 8:
                            # door类，RGB为 黄绿色
                            Empty_array[w, h, 0] = 127
                            Empty_array[w, h, 1] = 255
                            Empty_array[w, h, 2] = 0
                        elif row_Frame[w * Height + h] == 9:
                            # floor类，RGB为 白杏仁
                            Empty_array[w, h, 0] = 255
                            Empty_array[w, h, 1] = 235
                            Empty_array[w, h, 2] = 205
                        elif row_Frame[w * Height + h] == 10:
                            # sofa类，RGB为 棕色
                            Empty_array[w, h, 0] = 128
                            Empty_array[w, h, 1] = 42
                            Empty_array[w, h, 2] = 42
                        elif row_Frame[w * Height + h] == 11:
                            # table类，RGB为 淡黄
                            Empty_array[w, h, 0] = 245
                            Empty_array[w, h, 1] = 222
                            Empty_array[w, h, 2] = 179
                        elif row_Frame[w * Height + h] == 12:
                            # wall类，RGB为 天蓝色
                            Empty_array[w, h, 0] = 240
                            Empty_array[w, h, 1] = 255
                            Empty_array[w, h, 2] = 255
                        else:
                            # windows类，RGB为 黄色
                            Empty_array[w, h, 0] = 222
                            Empty_array[w, h, 1] = 255
                            Empty_array[w, h, 2] = 0
                # print(Empty_array.shape)
                # 将数组转化为图片
                img = Image.fromarray(Empty_array).convert('RGB').rotate(90)
                # 将数组保存为图片
                print(f'{index}.png 保存完成')
                img.save(eval_path + f'{index}.png')
            index += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
