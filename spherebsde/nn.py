import torch
import torch.nn as nn
import spherebsde.utils as utils
import time

class Beehive(nn.Module):
    def __init__(self, d, u0, t:torch.tensor, D:torch.tensor, data_gen, N, xb) -> None:
        super().__init__()
        self.d = d
        self.u0 = u0
        self.t = t
        self.D = D
        self.data_gen = data_gen
        self.N = N
        self.xb = xb
        self.dt = t / N
        self.fix_x = nn.Parameter(torch.tensor([1.,0,0]),requires_grad=False)

    def go_back(self, xt):
        xb, nb, in_ornot = self.xb(xt)
        xt[~in_ornot] = xt[~in_ornot] + 2*((xb[~in_ornot]-xt[~in_ornot])*(-nb[~in_ornot])).sum(dim=1,keepdim=True)*(-nb[~in_ornot])
        return xt, nb, ~in_ornot

class EBeehive(Beehive):
    def __init__(self, d, u0, t, D, data_gen, N, xb):
        super().__init__(d, u0, t, D, data_gen, N, xb)
        self.u = utils.FNN(d,1,64,3)
        self.grad_u = nn.ModuleList([utils.FNN(d,d,64,3) for _ in range(N)])
    
    def forward(self, batch):
        xt = self.data_gen(batch)
        dBt = torch.randn([self.N,batch,self.d],device=xt.device)*torch.sqrt(self.dt)
        grad_trans = torch.eye(self.d).expand(batch,self.d,self.d).clone().to(device=xt.device)
        u_pre = self.u(xt)
        for i in range(self.N):
            grad_u = torch.bmm(self.grad_u[i](xt).unsqueeze(1),grad_trans.clone()).squeeze(1)
            u_pre = u_pre + torch.sqrt(2*self.D)*(grad_u*dBt[i]).sum(dim=1,keepdim=True)
            xt, grad_trans = self.sde_euclidean(xt,grad_trans,dBt[i])
        u_rel = self.u0(xt)
        return u_pre, u_rel
    
    def sde_euclidean(self, xt, grad_trans, dBt):
        dXt = torch.sqrt(2*self.D) * dBt
        xt = xt + torch.bmm(grad_trans,dXt.unsqueeze(-1)).squeeze(-1)
        xt, nb, trans = self.go_back(xt)
        grad_transi = - 2*(-nb[trans].unsqueeze(-1))*(-nb[trans].unsqueeze(1))
        grad_trans[trans] = grad_trans[trans] + torch.bmm(grad_transi,grad_trans[trans])
        return xt, grad_trans

class SBeehive(Beehive):
    def __init__(self, u0, t, D, data_gen, N, xb):
        super().__init__(d=3, u0=u0, t=t, D=D, data_gen=data_gen, N=N, xb=xb)
        self.grad_eucl_polar = nn.Parameter(torch.tensor([[0.,0.],[0.,1.],[-1.,0.]]),requires_grad=False)
        self.u = utils.FNN(3,1,64,3)
        self.grad_u = nn.ModuleList([utils.FNN(3,3,64,3) for _ in range(N)])

    def forward(self, batch):
        xt = self.data_gen(batch)
        xt_in = xt.clone()
        dBt = torch.randn([self.N,batch,2]) * torch.sqrt(self.dt)
        grad_trans = torch.eye(3).expand(batch,3,3).clone()
        u_pre = self.u(xt)
        for i in range(self.N):
            x_polar = utils.polar_corr(xt)
            x_polar[:,0] -= torch.pi/2
            T_inv = utils.T_inverse(x_polar)
            grad_u = (torch.bmm(torch.bmm(self.grad_u[i](xt_in).unsqueeze(1),grad_trans.clone()),T_inv).transpose(1,2)*self.grad_eucl_polar).sum(dim=1)
            u_pre = u_pre + torch.sqrt(2*self.D)*(grad_u*dBt[i].clone()).sum(dim=1,keepdim=True)
            xt, xt_in, grad_trans = self.sde_sphere(xt,xt_in,grad_trans,T_inv,dBt[i])
        u_rel = self.u0(xt_in)
        return u_pre, u_rel
    
    def sde_sphere(self, xt, xt_in, grad_trans, T_inv, dBt):
        dXt = torch.sqrt(2*self.D) * dBt
        dXt[:,0] += torch.pi/2
        dXt = utils.transform_x(dXt) - self.fix_x
        dXt = torch.bmm(T_inv,dXt.unsqueeze(-1))
        xt = xt + dXt.squeeze(-1)
        xt_in = xt_in + torch.bmm(grad_trans,dXt).squeeze(-1)
        xt_in, nb, trans = self.go_back(xt_in)
        grad_transi = - 2*(-nb[trans]).unsqueeze(-1)*(-nb[trans]).unsqueeze(1)
        grad_trans[trans] = grad_trans[trans] + torch.bmm(grad_transi,grad_trans[trans])
        return xt, xt_in, grad_trans

class GrowthSBeehive(Beehive):
    def __init__(self, rmin, rmax, lamb, mu, u0, t, D, data_gen, N, xb, eq_type='backward', mc_size=10**4, P=128):
        super().__init__(d=3, u0=u0, t=t, D=D, data_gen=data_gen, N=N, xb=xb)
        self.grad_eucl_polar = nn.Parameter(torch.tensor([[0.,0.],[0.,1.],[-1.,0.]]),requires_grad=False)
        self.rmin = rmin
        self.rmax = rmax
        self.gap_r = rmax - rmin
        self.mc_size = mc_size
        self.P = P
        self.eq_type = eq_type
        self.r_all = nn.Parameter(torch.arange(rmin,rmax+1,1),requires_grad=False)
        self.k_all = nn.Parameter(torch.arange(1,self.gap_r+1,1),requires_grad=False)
        alpha = lamb if eq_type == 'backward' else lambda r,k:mu(r+k,k)
        beta = mu if eq_type == 'backward' else lambda r,k:lamb(r-k,k)
        self.alpha = nn.Parameter(torch.stack([alpha(self.r_all[i],self.k_all) for i in range(self.gap_r+1)],dim=0),requires_grad=False)
        self.beta = nn.Parameter(torch.stack([beta(self.r_all[i],self.k_all) for i in range(self.gap_r+1)],dim=0),requires_grad=False)
        self.lamb = nn.Parameter(torch.stack([lamb(self.r_all[i],self.k_all) for i in range(self.gap_r+1)],dim=0),requires_grad=False)
        self.mu = nn.Parameter(torch.stack([mu(self.r_all[i],self.k_all) for i in range(self.gap_r+1)],dim=0),requires_grad=False)
        jump_measure = torch.cat([self.alpha,self.beta],dim=1).cumsum(dim=1)
        self.cr = nn.Parameter(jump_measure[:,-1:],requires_grad=False)
        self.jump_measure = nn.Parameter(jump_measure / jump_measure[:,-1:],requires_grad=False)
        self.cfr = nn.Parameter(self.lamb.sum(dim=1,keepdim=True) + self.mu.sum(dim=1,keepdim=True) - self.cr,requires_grad=False)

    def jump(self, rt):
        B = rt.shape[0]
        jump_ornot = torch.rand([B]).to(rt.device) < self.cr[rt-self.rmin,0]*self.dt
        return self.jump_size(rt) * jump_ornot

    def jump_size(self, r):
        B = r.shape[0]
        jump_size_ind = self.gap_r*2 - (torch.rand([B,1]).to(r.device) < self.jump_measure[r-self.rmin]).sum(dim=1)
        jump_size = (jump_size_ind < self.gap_r) * (jump_size_ind+1)\
            - (jump_size_ind >= self.gap_r) * (jump_size_ind-self.gap_r+1)
        return jump_size
    
    def go_back(self, rt, xt):
        xb, nb, in_ornot = self.xb(rt,xt)
        xt[~in_ornot] = xt[~in_ornot] + 2*((xb[~in_ornot]-xt[~in_ornot])*(-nb[~in_ornot])).sum(dim=1,keepdim=True)*(-nb[~in_ornot])
        return xt, nb, ~in_ornot
    
class MKCapture(GrowthSBeehive):
    def __init__(self, rmin, rmax, lamb, mu, u0, t, Dx, Dy, data_gen, N, xb, capture, mc_size=10**4, P=128):
        super().__init__(rmin, rmax, lamb, mu, u0, t, Dx, data_gen, N, xb, 'forward', mc_size, P)
        self.Dy = Dy
        self.capture = capture
        self.u = nn.Parameter(torch.rand(1),requires_grad=True)
        self.grad_u = nn.ModuleList([utils.TNN(self.gap_r+1,6,6,P,64,3) for _ in range(N)])
        self.jump_x = nn.ModuleList([utils.FNN(6,P,64,3) for _ in range(N)])
        self.jump_r = nn.Parameter(torch.randn([self.gap_r+1,P])/torch.sqrt(torch.tensor(P)),requires_grad=True)
        self.jump_l = nn.Parameter(torch.randn([2,self.gap_r,P])/torch.sqrt(torch.tensor(P)),requires_grad=True)

    def forward(self, batch):
        rt,xt,yt_in = self.data_gen(batch)
        xt_in = xt.clone()
        dBxt = torch.randn([self.N,batch,2]).to(xt.device) * torch.sqrt(self.dt)
        dByt = torch.randn([self.N,batch,3]).to(xt.device) * torch.sqrt(self.dt)
        grad_trans_x = torch.eye(3).expand(batch,3,3).clone().to(xt.device)
        grad_trans_y = torch.eye(3).expand(batch,3,3).clone().to(xt.device)
        mc_samples = self.jump_size(self.r_all.expand(self.mc_size,self.gap_r+1).reshape(-1))
        mc_jump = self.jump_l[0,mc_samples.abs()-1] * (mc_samples.unsqueeze(1)>0) + self.jump_l[1,mc_samples.abs()-1] * (mc_samples.unsqueeze(1)<0)
        mc_jump = mc_jump.reshape(self.mc_size,self.gap_r+1,self.P).mean(dim=0) * self.cr
        u_pre = torch.ones([batch,1]).to(xt.device)*self.u
        fun = torch.zeros([batch,1]).to(xt.device)
        u_rel = torch.zeros([batch,1]).to(xt.device)
        run = torch.ones(batch).bool().to(xt.device)
        for i in range(self.N):
            drt = self.jump(rt)
            x_polar = utils.polar_corr(xt)
            x_polar[:,0] -= torch.pi/2
            T_inv = utils.T_inverse(x_polar)
            grad_u = self.grad_u[i](rt-self.rmin,torch.cat([xt_in,yt_in],dim=1))
            grad_ux = (torch.bmm(torch.bmm(grad_u[:,:3].unsqueeze(1),grad_trans_x.clone()),T_inv).transpose(1,2)*self.grad_eucl_polar).sum(dim=1)
            grad_uy = torch.bmm(grad_u[:,3:].unsqueeze(1),grad_trans_y.clone()).squeeze(1)
            jump_l = torch.zeros([batch,self.P]).to(xt.device)
            jump_l[drt.abs()>0] = jump_l[drt.abs()>0] + self.jump_l[0,drt[drt.abs()>0].abs()-1] * (drt[drt.abs()>0].unsqueeze(1) > 0)\
             + self.jump_l[1,drt[drt.abs()>0].abs()-1] * (drt[drt.abs()>0].unsqueeze(1) < 0)
            jump = (self.jump_x[i](torch.cat([xt_in,yt_in],dim=1)) * self.jump_r[rt-self.rmin] * (jump_l - mc_jump[rt-self.rmin]*self.dt)).sum(dim=1,keepdim=True)
            u_pre = u_pre + run.unsqueeze(1)*torch.exp(-fun)*(torch.sqrt(2*self.D)/rt.unsqueeze(-1)*(grad_ux*dBxt[i]).sum(dim=1,keepdim=True) + torch.sqrt(2*self.Dy)*(grad_uy*dByt[i]).sum(dim=1,keepdim=True) + jump)
            fun = fun + self.cfr[rt-self.rmin]*self.dt
            rt, xt, xt_in, yt_in, grad_trans_x, grad_trans_y = self.sde_mkcapture(rt,xt,xt_in,yt_in,grad_trans_x,grad_trans_y,T_inv,drt,dBxt[i],dByt[i])
            run = run * (~self.capture(rt,xt_in,yt_in))
        u_rel[run] = u_rel[run] + (self.u0(rt,xt_in,yt_in)*torch.exp(-fun))[run]
        return u_pre, u_rel
    
    def sde_mkcapture(self, rt, xt, xt_in, yt_in, grad_trans_x, grad_trans_y, T_inv, drt, dBxt, dByt):
        dXt = torch.sqrt(2*self.D)/rt.unsqueeze(-1) * dBxt
        dXt[:,0] += torch.pi/2
        dXt = utils.transform_x(dXt) - self.fix_x
        dXt = torch.bmm(T_inv,dXt.unsqueeze(-1))
        xt = xt + dXt.squeeze(-1)
        xt_in = xt_in + torch.bmm(grad_trans_x,dXt).squeeze(-1)
        dYt = torch.sqrt(2*self.Dy) * dByt
        yt_in = yt_in + torch.bmm(grad_trans_y,dYt.unsqueeze(-1)).squeeze(-1)
        rt = rt + drt
        xt_in, nbx, transx, yt_in, nby, transy = self.go_back(rt,xt_in,yt_in)
        grad_transxi = - 2*(-nbx[transx]).unsqueeze(-1)*(-nbx[transx]).unsqueeze(1)
        grad_trans_x[transx] = grad_trans_x[transx] + torch.bmm(grad_transxi,grad_trans_x[transx])
        grad_transyi = - 2*(-nby[transy]).unsqueeze(-1)*(-nby[transy]).unsqueeze(1)
        grad_trans_y[transy] = grad_trans_y[transy] + torch.bmm(grad_transyi,grad_trans_y[transy])
        return rt, xt, xt_in, yt_in, grad_trans_x, grad_trans_y
    
    def go_back(self, rt, xt, yt):
        xb, nxb, in_ornotx, yb, nyb, in_ornoty = self.xb(rt,xt,yt)
        xt[~in_ornotx] = xt[~in_ornotx] + 2*((xb[~in_ornotx]-xt[~in_ornotx])*(-nxb[~in_ornotx])).sum(dim=1,keepdim=True)*(-nxb[~in_ornotx])
        yt[~in_ornoty] = yt[~in_ornoty] + 2*((yb[~in_ornoty]-yt[~in_ornoty])*(-nyb[~in_ornoty])).sum(dim=1,keepdim=True)*(-nyb[~in_ornoty])
        return xt, nxb, ~in_ornotx, yt, nyb, ~in_ornoty

class GrowthRotation(GrowthSBeehive):
    def __init__(self, rmin, rmax, lamb, mu, u0, t, D, data_gen, N, xb, eq_type='backward', mc_size=10**4, P=128):
        super().__init__(rmin, rmax, lamb, mu, u0, t, D, data_gen, N, xb, eq_type, mc_size, P)
        self.u = utils.TNN(self.gap_r+1,3,1,P,64,3)
        self.grad_u = nn.ModuleList([utils.TNN(self.gap_r+1,3,3,P,64,3) for _ in range(N)])
        self.jump_x = nn.ModuleList([utils.FNN(3,P,64,3) for _ in range(N)])
        self.jump_r = nn.Parameter(torch.randn([self.gap_r+1,P])/torch.sqrt(torch.tensor(P)),requires_grad=True)
        self.jump_l = nn.Parameter(torch.randn([2,self.gap_r,P])/torch.sqrt(torch.tensor(P)),requires_grad=True)

    def forward(self, batch):
        rt,xt = self.data_gen(batch)
        xt_in = xt.clone()
        dBt = torch.randn([self.N,batch,2]).to(xt.device) * torch.sqrt(self.dt)
        grad_trans = torch.eye(3).expand(batch,3,3).clone().to(xt.device)
        mc_samples = self.jump_size(self.r_all.expand(self.mc_size,self.gap_r+1).reshape(-1))
        mc_jump = self.jump_l[0,mc_samples.abs()-1] * (mc_samples.unsqueeze(1)>0) + self.jump_l[1,mc_samples.abs()-1] * (mc_samples.unsqueeze(1)<0)
        mc_jump = mc_jump.reshape(self.mc_size,self.gap_r+1,self.P).mean(dim=0) * self.cr
        u_pre = self.u(rt-self.rmin,xt)
        fun = torch.zeros([batch,1]).to(xt.device)
        for i in range(self.N):
            drt = self.jump(rt)
            x_polar = utils.polar_corr(xt)
            x_polar[:,0] -= torch.pi/2
            T_inv = utils.T_inverse(x_polar)
            grad_u = (torch.bmm(torch.bmm(self.grad_u[i](rt-self.rmin,xt_in).unsqueeze(1),grad_trans.clone()),T_inv).transpose(1,2)*self.grad_eucl_polar).sum(dim=1)
            jump_l = torch.zeros([batch,self.P]).to(xt.device)
            jump_l[drt.abs()>0] = jump_l[drt.abs()>0] + self.jump_l[0,drt[drt.abs()>0].abs()-1] * (drt[drt.abs()>0].unsqueeze(1) > 0)\
             + self.jump_l[1,drt[drt.abs()>0].abs()-1] * (drt[drt.abs()>0].unsqueeze(1) < 0)
            jump = (self.jump_x[i](xt_in) * self.jump_r[rt-self.rmin] * (jump_l - mc_jump[rt-self.rmin]*self.dt)).sum(dim=1,keepdim=True)
            u_pre = u_pre + torch.exp(-fun)*(torch.sqrt(2*self.D)/rt.unsqueeze(-1)*(grad_u*dBt[i]).sum(dim=1,keepdim=True) + jump)
            fun = fun + self.cfr[rt-self.rmin]*self.dt
            rt, xt, xt_in, grad_trans = self.sde_growthrotation(rt,xt,xt_in,grad_trans,T_inv,drt,dBt[i])
        u_rel = self.u0(rt,xt_in)*torch.exp(-fun)
        return u_pre, u_rel
    
    def sde_growthrotation(self, rt, xt, xt_in, grad_trans, T_inv, drt, dBt):
        dXt = torch.sqrt(2*self.D)/rt.unsqueeze(-1) * dBt
        dXt[:,0] += torch.pi/2
        dXt = utils.transform_x(dXt) - self.fix_x
        dXt = torch.bmm(T_inv,dXt.unsqueeze(-1))
        xt = xt + dXt.squeeze(-1)
        xt_in = xt_in + torch.bmm(grad_trans,dXt).squeeze(-1)
        rt = rt + drt
        xt_in, nb, trans = self.go_back(rt,xt_in)  
        grad_transi = - 2*(-nb[trans]).unsqueeze(-1)*(-nb[trans]).unsqueeze(1)
        grad_trans[trans] = grad_trans[trans] + torch.bmm(grad_transi,grad_trans[trans])
        return rt, xt, xt_in, grad_trans

class GrowthRotationF(GrowthSBeehive):
    def __init__(self, rmin, rmax, lamb, mu, U, u0, t, D, data_gen, N, xb, eq_type='backward', mc_size=10**4, P=128):
        super().__init__(rmin, rmax, lamb, mu, u0, t, D, data_gen, N, xb, eq_type, mc_size, P)
        self.U = U
        self.u = utils.TNN(self.gap_r+1,4,2,P,64,3)
        self.grad_u = nn.ModuleList([utils.TNN(self.gap_r+1,4,6,P,64,3) for _ in range(N)])
        self.jump_x = nn.ModuleList([utils.FNN(4,2*P,64,3) for _ in range(N)])
        self.jump_r = nn.Parameter(torch.randn([self.gap_r+1,2*P])/torch.sqrt(torch.tensor(P)),requires_grad=True)
        self.jump_l = nn.Parameter(torch.randn([2,self.gap_r,2*P])/torch.sqrt(torch.tensor(P)),requires_grad=True)

    def forward(self, batch):
        p,rt,xt = self.data_gen(batch)
        xt_in = xt.clone()
        dBt = torch.randn([self.N,batch,2]).to(xt.device) * torch.sqrt(self.dt)
        grad_trans = torch.eye(3).expand(batch,3,3).clone().to(xt.device)
        mc_samples = self.jump_size(self.r_all.expand(self.mc_size,self.gap_r+1).reshape(-1))
        mc_jump = self.jump_l[0,mc_samples.abs()-1] * (mc_samples.unsqueeze(1)>0) + self.jump_l[1,mc_samples.abs()-1] * (mc_samples.unsqueeze(1)<0)
        mc_jump = mc_jump.reshape(self.mc_size,self.gap_r+1,2*self.P).mean(dim=0) * self.cr
        u_pre = self.u(rt-self.rmin,torch.cat([xt,p/40],dim=1))
        u_pre = torch.complex(u_pre[:,:1],u_pre[:,1:])
        fun = torch.zeros([batch,1]).to(xt.device)
        for i in range(self.N):
            drt = self.jump(rt)
            x_polar = utils.polar_corr(xt)
            x_polar[:,0] -= torch.pi/2
            T_inv = utils.T_inverse(x_polar)
            grad_u = self.grad_u[i](rt-self.rmin,torch.cat([xt_in,p/40],dim=1)).reshape([batch,2,3])
            grad_u = torch.bmm(torch.bmm(grad_u,grad_trans.clone()),T_inv)
            grad_u = torch.complex(grad_u[:,:1],grad_u[:,1:])
            grad_u = (grad_u.transpose(1,2)*self.grad_eucl_polar).sum(dim=1)
            jump_l = torch.zeros([batch,2*self.P]).to(xt.device)
            jump_l[drt.abs()>0] = jump_l[drt.abs()>0] + self.jump_l[0,drt[drt.abs()>0].abs()-1] * (drt[drt.abs()>0].unsqueeze(1) > 0)\
             + self.jump_l[1,drt[drt.abs()>0].abs()-1] * (drt[drt.abs()>0].unsqueeze(1) < 0)
            jump = (self.jump_x[i](torch.cat([xt_in,p/40],dim=1)) * self.jump_r[rt-self.rmin] * (jump_l - mc_jump[rt-self.rmin]*self.dt)).reshape([batch,2,self.P]).sum(dim=-1)
            jump = torch.complex(jump[:,:1],jump[:,1:])
            u_pre = u_pre + torch.exp(-fun)*(torch.sqrt(2*self.D)/rt.unsqueeze(-1)*(grad_u*dBt[i]).sum(dim=1,keepdim=True) + jump)
            rt, xt, xt_in, fun, grad_trans = self.sde_growthrotationF(rt,xt,xt_in,p,fun,grad_trans,T_inv,drt,dBt[i])
        u_rel = self.u0(p,rt,xt_in)*torch.exp(-fun)
        u_pre = u_pre
        return torch.cat([u_pre.real,u_pre.imag],dim=1), torch.cat([u_rel.real,u_rel.imag],dim=1)
    
    def sde_growthrotationF(self, rt, xt, xt_in, p, fun, grad_trans, T_inv, drt, dBt):
        fun = fun + (self.cfr[rt-self.rmin] + 1j*p*self.U(rt.unsqueeze(1)*xt))*self.dt
        dXt = torch.sqrt(2*self.D)/rt.unsqueeze(-1) * dBt
        dXt[:,0] += torch.pi/2
        dXt = utils.transform_x(dXt) - self.fix_x
        dXt = torch.bmm(T_inv,dXt.unsqueeze(-1))
        xt = xt + dXt.squeeze(-1)
        xt_in = xt_in + torch.bmm(grad_trans,dXt).squeeze(-1)
        rt = rt + drt
        xt_in, nb, trans = self.go_back(rt,xt_in)  
        grad_transi = - 2*(-nb[trans]).unsqueeze(-1)*(-nb[trans]).unsqueeze(1)
        grad_trans[trans] = grad_trans[trans] + torch.bmm(grad_transi,grad_trans[trans])
        return rt, xt, xt_in, fun, grad_trans

def train(model, params:dict, point=False):
    epoch = params['epoch']
    batch = params['batch']
    lr = params['lr']

    optim = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fun = nn.MSELoss()
    
    loss_values = torch.zeros(epoch)
    if point:
        res_values = torch.zeros(epoch)
    start = time.time()
    for i in range(epoch):
        model.train()
        optim.zero_grad()
        u_pre, u_rel = model(batch)
        loss = loss_fun(u_pre,u_rel)
        loss.backward()
        optim.step()

        model.eval()
        loss_values[i] = loss.item()
        if point:
            res_values[i] = model.u.detach()
            print('\r%5d/{}|{}{}|{:.2f}s  [Loss: %e, Result: %.6f]'.format(
                epoch,
                "#"*int((i+1)/epoch*50),
                " "*(50-int((i+1)/epoch*50)),
                time.time() - start) %
                (i+1,loss_values[i],res_values[i]), end = ' ', flush=True)
        else:
            print('\r%5d/{}|{}{}|{:.2f}s  [Loss: %e]'.format(
            epoch,
            "#"*int((i+1)/epoch*50),
            " "*(50-int((i+1)/epoch*50)),
            time.time() - start) %
            (i+1,loss_values[i]), end = ' ', flush=True)
    print("\nTraining has been completed.")
    if point:
        return loss_values, res_values
    else:
        return loss_values