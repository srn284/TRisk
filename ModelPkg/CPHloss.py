from torch import Tensor
import torch
import sklearn
import torch.nn as nn
from lifelines.utils import concordance_index



def cindex(pred, event, time):
#     event = torch.cat(event, dim=0).view(-1).numpy()
#     time = torch.cat(time, dim=0).view(-1).numpy()
#     partial_hazard = torch.exp(torch.cat(pred, dim=0).view(-1)).numpy()
    partial_hazard = torch.exp(pred).numpy()
    time = time.numpy()
    event = event.numpy()
    c_index = concordance_index(time, -1 * partial_hazard, event)

    return c_index



def roc_auc(logits, label, sig=True):
    sigm = nn.Sigmoid()
    if sig == True:
        output = sigm(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()

    tempprc = sklearn.metrics.roc_auc_score(label.numpy(), output.numpy())
    return tempprc, output, label

    
def precision(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
#     output=logits
    label, output=label.cpu(), output.detach().cpu()
    tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy())
    return tempprc, output, label

def precision_test(logits, label):
    sig = nn.Sigmoid()
    output=sig(logits)
    tempprc= sklearn.metrics.average_precision_score(label.numpy(),output.numpy())
    return tempprc, output, label

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    nll = - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum() + eps)
    return nll


def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.view(-1).sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        loss = cox_ph_loss(log_h, durations, events)
        #         print(loss)
        return loss