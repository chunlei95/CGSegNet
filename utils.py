import torch


class SearchBest(object):
    def __init__(self, min_delta=0):
        super(SearchBest, self).__init__()
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, val_loss, logger):
        if self.best is None:
            self.best = val_loss
        elif self.best - val_loss > self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info('performance reducing: {}'.format(self.counter))


# noinspection SpellCheckingInspection
def valid(val_loader, net, criterions: list, criterion_weights=None, device=None):
    if criterion_weights is None:
        criterion_weights = []
        for i in range(len(criterions)):
            criterion_weights.append(1)
    total_loss = 0.0
    # 前几次训练没有在验证的时候添加此段代码，导致训练结果严重不对经
    net.eval()
    for image, mask in val_loader:
        image = image.to(device)
        mask = mask.to(device).unsqueeze(1).to(torch.float32)
        with torch.no_grad():
            predict_mask = net(image)
            epoch_loss = 0.0
            for criterion, weight in zip(criterions, criterion_weights):
                loss = criterion(predict_mask, mask)
                epoch_loss += loss.item()
            total_loss += epoch_loss
    return total_loss / len(val_loader)
