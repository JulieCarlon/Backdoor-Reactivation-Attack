import torch
import math

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = I - unit / n
    H = H.to(K.device)
    return torch.matmul(torch.matmul(H, K), H)

def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist.item())
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def linear_HSIC(X, Y):
    if len(X.shape) > 2:
        X = X.view(X.shape[0], -1)
    if len(Y.shape) > 2:
        Y = Y.view(Y.shape[0], -1)
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))

def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# 使用方法：


if __name__=='__main__':
    set_seed(0)  # 将随机种子设为0
    X = torch.randn(100, 64)
    Y = torch.randn(100, 64)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))

