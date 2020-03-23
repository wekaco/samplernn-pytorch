import torch
from torch import nn
import numpy as np


EPSILON = 1e-2

def linear_quantize(samples, q_levels):
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return samples.long()

def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1

def q_zero(q_levels):
    return q_levels // 2



def mu_quantize(samples, q_levels=256.):
    """
    From Joao (+cj)
    x should be normalized between -1 and 1
    Converts an array according to mu-law and discretizes it
    Note:
        mu2linear(linear2mu(x)) != x
        Because we are compressing to 8 bits here.
        They will sound pretty much the same, though.
    """
    x = samples.clone()
    mu = q_levels - 1.
    x_mu = torch.sign(x) * np.log(1. + mu*torch.abs(x))/np.log(1. + mu)
    x_mu = ((x_mu + 1)/2 * mu)
    x_mu -= EPSILON
    x_mu += EPSILON / 2
    return x_mu.long()


def mu_dequantize(samples, q_levels=256.):
    """
    From Joao with modifications (+cj)
    Converts an integer array from mu to linear
    For important notes and usage see: linear2mu
    """

    mu = float(q_levels) - 1.
    x = samples.clone().float()
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return torch.sign(y) * (1./mu) * ((1. + mu)**torch.abs(y) - 1.)

