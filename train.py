from dataset import Dataset
from model import Model

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

import config

import argparse

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True


# 参数初始化
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('3dgnn')
    # 指定循环次数
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epoch')
    # 指定批量大小
    parser.add_argument('--batchsize', type=int, default=2, help='batch size in training')
    # 修改default,可以使用训练好的参数
    parser.add_argument('--pretrain', type=str, default=None, help='Direction for pretrained weight')
    # 指定GPU,一般从0开始编号
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    return parser.parse_args()


def main(args):
    # 指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 指定日志文件的存放位置
    logger = logging.getLogger('3dgnn')
    log_path = './train/' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H')).replace(' ', '/') + '/'
    print('日志路径:', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(log_path + 'save/')
    hdlr = logging.FileHandler(log_path + 'log.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    logger.info("加载数据...")
    print("加载数据...")

    # 创建字典，将标签和数字
    label_to_idx = {'<UNK>': 0, 'beam': 1, 'board': 2, 'bookcase': 3, 'ceiling': 4, 'chair': 5, 'clutter': 6,
                    'column': 7,
                    'door': 8, 'floor': 9, 'sofa': 10, 'table': 11, 'wall': 12, 'window': 13}

    idx_to_label = {0: '<UNK>', 1: 'beam', 2: 'board', 3: 'bookcase', 4: 'ceiling', 5: 'chair', 6: 'clutter',
                    7: 'column',
                    8: 'door', 9: 'floor', 10: 'sofa', 11: 'table', 12: 'wall', 13: 'window'}

    # config在该文件夹下有相应的python源代码，Dataset指定了相应的训练集
    # dataset_tr 是训练集
    dataset_tr = Dataset(flip_prob=config.flip_prob, crop_type='Random', crop_size=config.crop_size)
    dataloader_tr = DataLoader(dataset_tr, batch_size=args.batchsize, shuffle=True, num_workers=config.workers_tr,
                               drop_last=True, pin_memory=True)

    cv2.setNumThreads(config.workers_tr)

    # 日志文件需要添加的信息
    logger.info("准备模型...")
    print("准备模型...")

    # 模型初始化
    model = Model(config.nclasses, config.mlp_num_layers, config.use_gpu)
    loss = nn.NLLLoss(reduce=not config.use_bootstrap_loss, weight=torch.FloatTensor(config.class_weights))

    # dim表示维度,dim=0,表示行,dim=1,表示列
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    # 使用cuda加速
    if config.use_gpu:
        print(config.use_gpu)
        model = model.cuda()
        loss = loss.cuda()
        softmax = softmax.cuda()
        log_softmax = log_softmax.cuda()

    # 优化器,选择Adam
    optimizer = torch.optim.Adam([{'params': model.decoder.parameters()},
                                  {'params': model.gnn.parameters(), 'lr': config.gnn_initial_lr}],
                                 lr=config.base_initial_lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay)

    # 学习率调整策略，exp指指数衰减调整，plateau指自适应调整
    if config.lr_schedule_type == 'exp':
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), config.lr_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif config.lr_schedule_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.lr_decay,
                                                               patience=config.lr_patience)
    else:
        print('错误的调度程序')
        exit(1)

    # 记录训练参数数量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("可训练参数的数量: %d", params)

    # 得到目前的学习率
    def get_current_learning_rates():
        learning_rates = []
        for param_group in optimizer.param_groups:
            learning_rates.append(param_group['lr'])
        return learning_rates

    # 评估/预测,对输入数据进行评估/预测
    def eval_set(dataloader):
        model.eval()
        # torch.no_grad()函数使得程序不计算梯度,只进行前向传播,用在预测中正合适
        with torch.no_grad():
            loss_sum = 0.0

            # 混淆矩阵
            confusion_matrix = torch.cuda.FloatTensor(np.zeros(14 ** 2))

            # tqdm是进度条模块
            for batch_idx, rgbd_label_xy in enumerate(dataloader):
                # torch.Tensor 类型 [2, 640, 480, 6] 对应数据集中的 rgb_hha 矩阵
                # 第一个2是batch大小，640和480是宽和高，第二个6是因为其为rgb与hha的拼接
                x = rgbd_label_xy[0]

                # torch.Tensor 类型 [2, 640, 480, 2] 对应数据集中的 xy 矩阵
                # 第一个2是batch大小，640和480是宽和高，2是因为 xy 数据集只有两个维度，且为全零矩阵
                xy = rgbd_label_xy[2]

                # torch.Tensor 类型 [2, 640, 480] 第一个3是batch大小，640和480是宽和高
                target = rgbd_label_xy[1].long()
                x = x.float()
                xy = xy.float()

                # permute函数用于转换Tensor的维度，contiguous()使得内存是连续的
                input = x.permute(0, 3, 1, 2).contiguous()
                xy = xy.permute(0, 3, 1, 2).contiguous()
                if config.use_gpu:
                    input = input.cuda()
                    xy = xy.cuda()
                    target = target.cuda()

                # 经过网络，计算输出, 维度为([2, 14, 640, 480])
                output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy,
                               use_gnn=config.use_gnn)

                # config.use_bootstrap_loss为False
                if config.use_bootstrap_loss:
                    loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                    topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                               int((config.crop_size ** 2) * config.bootstrap_rate))
                    loss_ = torch.mean(topk)
                # log_softmax在softmax的结果上再做多一次log运算
                else:
                    loss_ = loss.forward(log_softmax(output.float()), target)
                loss_sum += loss_

                # pred维度为 ([2, 640, 480, 14]), 连续内存
                pred = output.permute(0, 2, 3, 1).contiguous()
                # 此时pred维度为 ([614400, 14]), 其中614400=2*640*480  config.nclasses=14
                pred = pred.view(-1, config.nclasses)
                # 每一行进行softmax运算，相当于对每一个像素的分类进行softmax运算
                # [614400, 14]
                pred = softmax(pred)
                # pred_max_val, pred_arg_max都是614400维，分别存储每个像素最大的分类值及分类
                pred_max_val, pred_arg_max = pred.max(1)  # 返回最大值和相应坐标  行级别 每行一个  614400个最大值

                # pairs为614400维
                pairs = target.view(-1) * 14 + pred_arg_max.view(-1)  # [ 0]

                # 计算混淆矩阵
                for i in range(14 ** 2):
                    cumu = pairs.eq(i).float().sum()
                    confusion_matrix[i] += cumu.item()

            loss_sum /= len(dataloader)

            confusion_matrix = confusion_matrix.cpu().numpy().reshape((14, 14))
            class_iou = np.zeros(14)
            confusion_matrix[0, :] = np.zeros(14)
            confusion_matrix[:, 0] = np.zeros(14)

            # 计算交并比
            for i in range(1, 14):
                class_iou[i] = confusion_matrix[i, i] / (
                        np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i])

        return loss_sum.item(), class_iou, confusion_matrix

    # Training parameter 训练参数
    model_to_load = args.pretrain  # 可以设置从哪个开始训练
    logger.info("训练轮数: %d", args.num_epochs)
    print("训练轮数: %d" % args.num_epochs)
    interval_to_show = 100

    train_losses = []
    eval_losses = []

    # 判断使用原来训练过的模型参数，还是从零开始训练
    if model_to_load:
        logger.info("加载旧模型...")
        print("加载旧模型...")
        model.load_state_dict(torch.load(model_to_load))
    else:
        logger.info("从头开始训练...")
        print("从头开始训练...")

    # Training
    # 循环训练，range范围是最后一个参数-1，所以想要实现指定次数，需要+1
    for epoch in range(1, args.num_epochs + 1):
        batch_loss_avg = 0

        # 学习率更新参数
        if config.lr_schedule_type == 'exp':
            scheduler.step(epoch)

        # 训练过程
        for batch_idx, rgbd_label_xy in enumerate(dataloader_tr):
            x = rgbd_label_xy[0]
            target = rgbd_label_xy[1].long()
            xy = rgbd_label_xy[2]

            x = x.float()
            xy = xy.float()

            input = x.permute(0, 3, 1, 2).contiguous()
            input = input.type(torch.FloatTensor)

            if config.use_gpu:
                input = input.cuda()
                xy = xy.cuda()
                target = target.cuda()

            xy = xy.permute(0, 3, 1, 2).contiguous()

            optimizer.zero_grad()
            model.train()
            # torch.Size([2, 14, 640, 480])
            output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy,
                           use_gnn=config.use_gnn)

            # config.use_bootstrap_loss=False
            if config.use_bootstrap_loss:
                loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                           int((config.crop_size ** 2) * config.bootstrap_rate))
                loss_ = torch.mean(topk)
            else:
                loss_ = loss.forward(log_softmax(output.float()), target)

            loss_.backward()
            optimizer.step()

            batch_loss_avg += loss_.item()

            if batch_idx % interval_to_show == 0 and batch_idx > 0:
                batch_loss_avg /= interval_to_show
                train_losses.append(batch_loss_avg)
                logger.info("Epoch:{}, Batch:{}, loss average:{}".format(epoch, batch_idx, batch_loss_avg))
                print('\rEpoch:{}, Batch:{}, loss average:{}'.format(epoch, batch_idx, batch_loss_avg))
                batch_loss_avg = 0

        # 训练结束，后续保存参数，并进行测试
        batch_idx = len(dataloader_tr)
        logger.info("Epoch:{}, Batch:{} 保存...".format(epoch, batch_idx))

        # 保存模型参数
        torch.save(model.state_dict(), log_path + '/save/' + 'checkpoint_' + str(epoch) + '.pth')

        # Evaluation

        # 每一次训练完以后，用测试集进行测试，看看分类效果
        eval_loss, class_iou, confusion_matrix = eval_set(dataloader_tr)
        eval_losses.append(eval_loss)

        # 另一种学习率更新的情况
        # if config.lr_schedule_type == 'plateau':
        #   scheduler.step(eval_loss)

        print('Learning ...')
        logger.info("Epoch{} 当前学习率: {}", epoch, batch_idx, get_current_learning_rates()[0])
        print('Epoch{} 当前学习率: {}'.format(epoch, get_current_learning_rates()[0]))

        logger.info("Epoch{} GNN学习率: %s", epoch, batch_idx, get_current_learning_rates()[1])
        print('Epoch{} GNN学习率: {}'.format(epoch, get_current_learning_rates()[1]))

        logger.info("Epoch{} ,Batch:{} 评价损失: %s", epoch, batch_idx, eval_loss)
        print('Epoch{} 评价损失: {}'.format(epoch, eval_loss))

        logger.info("Epoch{} Batch:{} 类别 IoU:", epoch, batch_idx)
        print('Epoch{} Class IoU:'.format(epoch))

        for cl in range(14):
            logger.info("%+10s: %-10s" % (idx_to_label[cl], class_iou[cl]))
            print('{}:{}'.format(idx_to_label[cl], class_iou[cl]))

        logger.info("平均 IoU: %s", np.mean(class_iou[1:]))
        print("平均 IoU: %.2f" % np.mean(class_iou[1:]))

        logger.info("Epoch{} Batch:{}  Confusion matrix:", epoch, batch_idx)
        logger.info(confusion_matrix)

    # 所有循环都结束了，此时在日志文件进行记录
    logger.info("完成训练!")
    logger.info("保存最终模型...")
    print('保存最终模型...')
    # 保存模型参数
    torch.save(model.state_dict(), log_path + '/save/finish.pth')

    # 预测集（也是训练集），重新放入网络进行前向传播，评估损失和交并比
    eval_loss, class_iou, confusion_matrix = eval_set(dataloader_tr)

    # 日志文件输出
    logger.info("评价损失: %s", eval_loss)
    logger.info("类别 IoU:")
    # 14个分类，每一类的IoU（交并比）
    for cl in range(14):
        logger.info("%+10s: %-10s" % (idx_to_label[cl], class_iou[cl]))
    # 所有类的平均交并比（IoU）
    logger.info("平均 IOU: %s", np.mean(class_iou[1:]))


if __name__ == '__main__':
    args = parse_args()
    main(args)