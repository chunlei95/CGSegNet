import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalContrastiveLoss(nn.Module):
    """全局对比损失

    """

    def __init__(self):
        super(GlobalContrastiveLoss, self).__init__()

    def forward(self, x):
        pass


class DenseContrastiveLoss(nn.Module):
    """局部对比损失

    """

    def __init__(self):
        super(DenseContrastiveLoss, self).__init__()

    def forward(self, x):
        pass


class DiceLoss(nn.Module):
    """todo 这个代码是从别的论文中复制过来的，但是感觉有问题

    """

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets):
        # 样本数
        N = preds.size(0)
        # 通道数
        C = preds.size(1)

        # 得到预测的概率值
        # P = F.softmax(preds, dim=1)
        P = preds
        # 定义平滑系数，值设置为1e-5
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets.to(torch.int64), 1.)

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(
            dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        # print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
        den = num + self.alpha * torch.sum(FP.transpose(0, 1).reshape(C, -1), dim=(1)).float() + self.beta * torch.sum(
            FN.transpose(0, 1).reshape(C, -1), dim=(1)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss

        loss = 1 - dice
        loss = loss.sum()

        # 控制是否需要对结果求平均，多分类需要对结果除以通道数求平均
        if self.size_average:
            loss /= C

        return loss


class ContrastiveLoss(nn.Module):
    """对比损失

    """

    def __init__(self, batch_size, device='cpu', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer('temperature', torch.tensor(temperature).to(device))
        self.register_buffer('negatives_mask',
                             (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool).to(device)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representation = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representation.unsqueeze(1), representation.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, ~self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = - torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


if __name__ == '__main__':
    # todo 验证一下损失函数是否正确
    pass
