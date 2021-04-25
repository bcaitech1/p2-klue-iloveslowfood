import torch
from torch import nn
from torch.nn import functional as F
from config import Loss, Config


def get_criterion(type: str = Loss.CE):
    """지정한 type에 따른 Loss 함수를 리턴

    Args:
        type (str, optional): loss 종류를 설정. 크로스엔트로피('crossentropyloss'), 레이블스무딩('labelsmoothingLoss') 지원. Defaults to 'crossentropyloss'.

    Returns:
        torch.nn: loss 함수
    """
    if type == Loss.CE:
        criterion = nn.CrossEntropyLoss()
    elif type == Loss.LS:
        criterion = LabelSmoothingLoss(classes=Config.NumClasses)
    return criterion


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.2, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
