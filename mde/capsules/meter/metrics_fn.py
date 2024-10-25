import torch
import numpy as np


def mae(out, pout, mask):
    return (out[mask] - pout[mask]).abs().sum(), mask.float().sum()

def mse(out, pout, mask):
    return ((out[mask] - pout[mask]) ** 2).sum(), mask.float().sum()


def rmse(out, pout, mask):
    enum, denum = mse(out, pout, mask)
    if denum > 0:
        return torch.sqrt(enum / denum)
    else:
        return 0, 0


def absrel(out, pout, mask):
    return ((pout[mask] - out[mask]).abs() / out[mask]).sum(), mask.float().sum()

def sqrel(out, pout, mask):
    return ((pout[mask] - out[mask])**2 /  out[mask]).sum(), mask.float().sum()

def _delta(out, pout, mask):
    return torch.max(out[mask] / pout[mask], pout[mask] / out[mask])

def d1(out, pout, mask):
    return (_delta(out, pout, mask) > 1.25).float().sum(), mask.float().sum()

def d2(out, pout, mask):
    return (_delta(out, pout, mask) > 1.25 ** 2).float().sum(), mask.float().sum()

def d3(out, pout, mask):
    return (_delta(out, pout, mask) > 1.25 ** 3).float().sum(), mask.float().sum()

def d102(out, pout, mask):
    return (_delta(out, pout, mask) > 1.02).float().sum(), mask.float().sum()

def d105(out, pout, mask):
    return (_delta(out, pout, mask) > 1.05).float().sum(), mask.float().sum()

def d110(out, pout, mask):
    return (_delta(out, pout, mask) > 1.10).float().sum(), mask.float().sum()



# def log10_core(one_pred, one_targ):
#     return (one_pred.log10() - one_targ.log10()).abs().sum()

# def a1_core(one_pred, one_targ):
#     thresh = torch.max((one_targ / one_pred), (one_pred / one_targ))
#     return (thresh < 1.25).sum()#.mean()

# def a2_core(one_pred, one_targ):
#     thresh = torch.max((one_targ / one_pred), (one_pred / one_targ))
#     return (thresh < 1.25 ** 2).sum()#.mean()

# def a3_core(one_pred, one_targ):
#     thresh = torch.max((one_targ / one_pred), (one_pred / one_targ))
#     return (thresh < 1.25 ** 3).sum()#.mean()

# def rel_core(one_pred, one_targ):
#     return ((one_pred - one_targ).abs() / one_targ).sum()

# def sqrel_core(one_pred, one_targ):
#     return ((one_pred - one_targ)**2 / one_targ).sum()

# def silog_core(one_pred, one_targ):
#     err = one_pred.log() - one_targ.log()
#     return torch.sqrt((err ** 2).mean() - err.mean() ** 2) * 100

# def delta_core(one_pred, one_targ):
#     return torch.max(one_pred / one_targ, one_targ / one_pred)

# def rmse_core(one_pred, one_gt):
#     return (one_pred - one_gt).pow(2).mean().sqrt() * one_pred.numel()

# def rmselog_core(one_pred, one_gt):
#     return (one_pred.log() - one_gt.log()).pow(2).mean().sqrt() * one_pred.numel()

# def irmse_core(one_pred, one_gt):
#     return ((1.0 / one_pred) - (1.0 / one_gt)).pow(2).mean().sqrt() * one_pred.numel()

# def mae_core(one_pred, one_gt):
#     return (one_pred - one_gt).abs().sum()

# def imae_core(one_pred, one_gt):
#     return ((1.0 / one_pred) - (1.0 / one_gt)).abs().sum()

# def d1_core(one_pred, one_targ):
#     return (delta_core(one_pred, one_targ) > 1.25).float().sum()

# def d102_core(one_pred, one_targ):
#     return (delta_core(one_pred, one_targ) > 1.02).float().sum()

# def d105_core(one_pred, one_targ):
#     return (delta_core(one_pred, one_targ) > 1.05).float().sum()

# def d110_core(one_pred, one_targ):
#     return (delta_core(one_pred, one_targ) > 1.10).float().sum()

# def d2_core(one_pred, one_targ):
#     return (delta_core(one_pred, one_targ) < 1.25 ** 2).float().sum()


# def d3_core(one_pred, one_targ):
#     return (delta_core(one_pred, one_targ) < 1.25 ** 3).float().sum()


