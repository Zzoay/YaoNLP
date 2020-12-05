
from torch import max
from torch.nn.functional import cross_entropy


def compute_acc(logit, y_gt):
    predicts = max(logit, 1)[1]
    corrects = (predicts.view(y_gt.size()).data == y_gt.data).float().sum()
    accuracy = 100.0 * float(corrects/len(y_gt))

    return accuracy