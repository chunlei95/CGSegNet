import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import transforms
from dataset import load_dataset
from logger import get_logger
from model.CGSegNet import CGSegNet
from model.CGSegNet import PatchEmbedding
from utils import valid, SearchBest

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args_parser.add_argument('--te', type=int, default=150, help='total epoch')
args_parser.add_argument('--ce', type=int, default=0, help='current epoch')
args_parser.add_argument('--bs', type=int, default=8, help='batch size')
args_parser.add_argument('--bw', type=float, default=0.8, help='weight of bce loss')
args_parser.add_argument('--dw', type=float, default=0.2, help='weight of dice loss')
args = args_parser.parse_args()

logger = get_logger('logs/v1.log')


# todo 这一部分是一个很难的问题，流程怎么样一定要弄好
# noinspection PyShadowingNames
def pre_train(net, train_loader, criterion, optimizer, current_epoch, total_epoch, device):
    """自监督预训练

    :param net:
    :param train_loader:
    :param criterion:
    :param optimizer:
    :param current_epoch:
    :param total_epoch:
    :param device:
    :return:
    """
    for i in range(current_epoch, total_epoch):
        for index, (x, y) in enumerate(train_loader):
            x = x.to(device)
            # 处理x，使其作为对比分支CNN编码器的输入x1
            # step1: 得到关于x的两个增强版本
            # step2: 将两个x输入到CNN编码器，进行对比训练
            # step3: 得到相关的负类样本
            # 处理x，使其作为生成分支transformer编码器的输入x2
            patch_embedding = PatchEmbedding()
            x_patch_embed = patch_embedding(x)
            # step1: 将x分割为patch，然后进行patch嵌入表示
            # step2: 将patch嵌入表示输入到transformer
            # 将x1, x2输入到模型中进行计算，得到需要的结果
            segment_result = net(x, x_patch_embed)
            # 根据模型返回的结果计算对比损失以及生成损失
            # 梯度反向传播
            # 调用优化器
            pass
    # 保存预训练后的模型参数
    pass


# noinspection PyShadowingNames
def train(net, train_loader, valid_loader, optimizer, criterion, current_epoch=0,
          total_epoch=100, device=None):
    """全监督方式

    :param net:
    :param train_loader:
    :param valid_loader:
    :param criterion:
    :param optimizer:
    :param current_epoch:
    :param total_epoch:
    :param device:
    :return:
    """
    net.to(device)
    criterion.to(device)
    train_losses = []
    val_losses = []
    save_last = {}
    save_best = {}
    loss_history = {}
    for i in range(current_epoch, total_epoch):
        total_loss = 0.0
        search_best = SearchBest()
        net.train()
        for index, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            segment_result = net(x)
            epoch_loss = criterion(segment_result.squeeze(), y.to(torch.float32))
            optimizer.zero_grad()
            # 梯度反向传播
            epoch_loss.backward()
            # 调用优化器
            optimizer.step()

            total_loss += epoch_loss.item()
            logger.info(
                'Epoch {}: Batch {}/{} train loss = {:.4f}'.format(i + 1, index + 1, len(train_loader),
                                                                   epoch_loss.item()))
        val_loss = valid(valid_loader, net, criterions=[criterion], device=device)
        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)
        logger.info('Epoch {}: train loss = {:.4f} val loss = {:.4f}'.format(i + 1, total_loss / len(train_loader),
                                                                             val_loss))
        search_best(val_loss, logger)
        if search_best.counter == 0:
            save_best['model_state_dict'] = net.state_dict()
            save_best['current_epoch'] = i + 1

    logger.info('end training!')

    save_last['model_state_dict'] = net.state_dict()
    save_last['optimizer_state_dict'] = optimizer.state_dict()
    save_last['last_epoch'] = total_epoch
    loss_history['train_loss_history'] = train_losses
    loss_history['val_loss_history'] = val_losses
    torch.save(save_last, 'pretrained/v1/last.pth')
    torch.save(save_best, 'pretrained/v1/best.pth')
    torch.save(loss_history, 'pretrained/v1/loss_history.pth')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    net = CGSegNet().to(device)
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.99))
    criterion = nn.BCELoss().to(device)
    train_loader, val_loader = load_dataset('B', batch_size=args.bs, train_transforms=train_trans,
                                            test_transforms=val_trans)
    train(net, train_loader, val_loader, optimizer, criterion, total_epoch=args.te, current_epoch=args.ce,
          device=device)
