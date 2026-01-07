import torch

def agreement_conv(k, q_global):
    return (k * q_global[:, :, None, None]).sum(dim=1, keepdim=True)


def agreement_dense(k, q_global):
    return (k * q_global).sum(dim=1, keepdim=True)
