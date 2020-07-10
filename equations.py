import torch

def h(x, eps):
    return torch.sign(x)*(torch.sqrt(torch.abs(x)+1)-1) + eps*x

def h_inv(x, eps):
    return torch.sign(x)*(((torch.sqrt(1+4*eps*(torch.abs(x)+1+eps))-1)/(2*eps))**2-1)


