from numpy import deprecate
import torch
import torch.nn as nn
from torch.nn import functional as F


def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()  # dim=-1
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def softmax_sigmoid_BCEloss(pred, targets):
    prob = torch.nn.functional.softmax(pred, dim=1)
    prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
    logit = torch.log((prob / (1 - prob)))
    loss_func = torch.nn.BCEWithLogitsLoss()
    return loss_func(logit, targets)


def norm_logits_BCEloss(pred, targets):
    loss_func = torch.nn.BCEWithLogitsLoss()
    return loss_func(pred, targets)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,  #0.25,
    gamma: float = 2,
    reduction: str = "mean",
):
    """
    support soft label
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction="none")
    # p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)
    p_t = torch.abs(targets - p)
    loss = ce_loss * (p_t**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@deprecate
def sigmoid_ASL_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    gamma_: float = 2,
    c: float = 0.05,
    reduction: str = "mean",
):
    """
    NOT support soft label
    """
    p = torch.sigmoid(inputs)
    neg_flag = (1 - targets).float()
    p = torch.clamp(p - neg_flag * c, 1e-9)

    ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction="none")
    p_pos = ((1 - p)**gamma) * targets + (p**gamma_) * (1 - targets)
    loss = ce_loss * p_pos

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def ranking_loss(y_pred, y_true, scale_=2.0, margin_=1):
    y_pred *= scale_
    y_true_ = y_true.float()
    tmp = margin_ - y_pred[:, None, :] + y_pred[:, :, None]
    partial_losses = torch.maximum(torch.zeros_like(tmp), tmp)
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    loss = torch.sum(loss, dim=-1)
    loss = torch.sum(loss, dim=-1)
    return torch.mean(loss)


class AsymmetricLoss_partial(nn.Module):
    def __init__(self,
                 gamma_neg=4,
                 gamma_pos=1,
                 clip=0.05,
                 eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss_partial, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, thresh_pos=0.9, thresh_neg=-0.9, if_partial=True):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        y_pos = (y > thresh_pos).float()
        y_neg = (y < thresh_neg).float()
        # Basic CE calculation
        los_pos = y_pos * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = y_neg * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y_pos
            pt1 = xs_neg * y_neg  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y_pos + self.gamma_neg * y_neg
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / x.shape[0] if if_partial else -loss.mean()


def dualcoop_loss(inputs, inputs_g, targets):
    """
    using official ASL loss.
    """
    loss_fun = AsymmetricLoss_partial(gamma_neg=2, gamma_pos=1, clip=0.05)

    return loss_fun(inputs, targets, thresh_pos=0.9,
                    thresh_neg=-0.9)  # + loss_fun(inputs_g, targets)


def ASL_loss(inputs, targets):
    """
    full label ASLOSS
    """
    loss_fun = AsymmetricLoss_partial(gamma_neg=2, gamma_pos=1, clip=0.05)

    return loss_fun(inputs,
                    targets,
                    thresh_pos=0.9,
                    thresh_neg=0.9,
                    if_partial=False)
