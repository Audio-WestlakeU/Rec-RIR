# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: loss functions.

import torch.nn as nn


class MSE_loss_complex(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, y, x):
        return (x - y).abs().pow(2).mean()


class RIMag_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        ret = (
            (output.real - target.real).abs()
            + (output.imag - target.imag).abs()
            + (output.abs() - target.abs()).abs()
        )
        return ret.mean()
