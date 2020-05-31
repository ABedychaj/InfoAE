import torch
from sklearn.metrics import normalized_mutual_info_score


class LossFunction:
    def __init__(self):
        pass

    def loss(self, **kwargs):
        raise NotImplementedError


class MutualInformationLoss(LossFunction):
    def __init__(self, cuda=False):
        super(MutualInformationLoss, self).__init__()
        self.cuda = cuda

    def loss(self, x, y):
        return 1 - normalized_mutual_info_score(torch.flatten(y), torch.flatten(x))
