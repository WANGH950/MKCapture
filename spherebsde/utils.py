import torch
import torch.nn as nn
from collections import OrderedDict

def grad(x: torch.tensor, y: torch.tensor):
    _,d = y.shape
    grad_y = []
    for i in range(d):
        ones = torch.zeros_like(y)
        ones[:,i] = 1
        grad_y.append(
            torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=ones,
                retain_graph=True,
                create_graph=True
            )[0].unsqueeze(1)
        )
    return torch.cat(grad_y,dim=1)

def transform_x(x):
    x_cos = torch.cos(x)
    x_sin = torch.sin(x)
    return torch.cat([x_sin[:,:1]*x_cos[:,1:],x_sin[:,:1]*x_sin[:,1:],x_cos[:,:1]],dim=1)

def grad_transform_x(x):
    x_cos = torch.cos(x)
    x_sin = torch.sin(x)
    x_transform_grad_theta = torch.cat([x_cos[:,:1]*x_cos[:,1:],x_cos[:,:1]*x_sin[:,1:],-x_sin[:,:1]],dim=1).unsqueeze(-1)
    x_transform_grad_phi = torch.cat([-x_sin[:,:1]*x_sin[:,1:],x_sin[:,:1]*x_cos[:,1:],x_cos[:,:1]*0],dim=1).unsqueeze(-1)
    return torch.cat([x_transform_grad_theta,x_transform_grad_phi],dim=2)

def polar_corr(x):
    theta = torch.pi/2 - torch.arctan(x[:,2:3]/x[:,:2].norm(dim=1,keepdim=True))
    phi = torch.pi/2 - torch.arctan(x[:,:1]/x[:,1:2]) + (x[:,1:2]<0)*torch.pi
    phi[torch.isnan(phi)] = 0.
    return torch.cat([theta,phi],dim=1)

def T_inverse(x_polar):
    B = x_polar.shape[0]
    trans1 = torch.cat([torch.cat([torch.cos(x_polar[:,:1]),torch.zeros([B,1]).to(x_polar.device),torch.sin(x_polar[:,:1])],dim=1).unsqueeze(1),torch.cat([torch.zeros([B,1]).to(x_polar.device),torch.ones([B,1]).to(x_polar.device),torch.zeros([B,1]).to(x_polar.device)],dim=1).unsqueeze(1),torch.cat([-torch.sin(x_polar[:,:1]),torch.zeros([B,1]).to(x_polar.device),torch.cos(x_polar[:,:1])],dim=1).unsqueeze(1)],dim=1)
    trans2 = torch.cat([torch.cat([torch.cos(x_polar[:,1:]),-torch.sin(x_polar[:,1:]),torch.zeros([B,1]).to(x_polar.device)],dim=1).unsqueeze(1),torch.cat([torch.sin(x_polar[:,1:]),torch.cos(x_polar[:,1:]),torch.zeros([B,1]).to(x_polar.device)],dim=1).unsqueeze(1),torch.cat([torch.zeros([B,1]).to(x_polar.device),torch.zeros([B,1]).to(x_polar.device),torch.ones([B,1]).to(x_polar.device)],dim=1).unsqueeze(1)],dim=1)
    return torch.bmm(trans2,trans1)

class FNN(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, num_layers=3) -> None:
        super(FNN, self).__init__()
        if num_layers == 0:
            self.fnn = nn.Linear(in_dim, out_dim)
        else:
            fnn_layers = [nn.Sequential(
                    nn.Linear(in_dim,hid_dim+out_dim),
                    nn.ReLU()
                )] + [nn.Sequential(
                    nn.Linear(hid_dim+out_dim,hid_dim+out_dim),
                    nn.ReLU()
                ) for _ in range(num_layers)] + [
                    nn.Linear(hid_dim+out_dim,out_dim)
                ]
            self.fnn = nn.Sequential(*fnn_layers)

    def forward(self, x):
        y = self.fnn(x)
        return y
    
class TNN(nn.Module):
    def __init__(self, dr, dx, out_dim=1, P=64, hid_dim=64, num_layers=3) -> None:
        super(TNN, self).__init__()
        self.dr = dr
        self.dx = dx
        self.out_dim = out_dim
        self.P = P
        self.fnn = FNN(dx, out_dim*P, hid_dim, num_layers)
        self.tr = nn.Parameter(torch.randn(dr,P)/torch.sqrt(torch.tensor(P)),requires_grad=True)

    def forward(self, r, x):
        B = r.shape[0]
        yx = self.fnn(x).reshape([B,self.P,self.out_dim])
        yr = self.tr[r].unsqueeze(-1)
        y = (yx*yr).sum(dim=1)
        return y