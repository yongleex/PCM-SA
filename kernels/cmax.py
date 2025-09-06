import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def estimateCMAX(evts, K=32, sigma2=1.0, n_iter=128, debug=False, **kwargs):
    def iwe_var(evts, v0, K, sigma2):
        # device = evts.device
        
        x, y, t, p = evts[:, 0], evts[:, 1], evts[:, 2], evts[:, 3]
        vx, vy = v0[:, 0], v0[:,1]

        t = t-torch.mean(t)
        
        x0, y0 = x - vx * t, y - vy * t  # (N,)
        
        px = torch.linspace(torch.min(x0).item(), torch.max(x0).item(), K, device=evts.device)
        py = torch.linspace(torch.min(y0).item(), torch.max(y0).item(), K, device=evts.device)
        px_g, py_g = torch.meshgrid(px, py, indexing='xy')
    
        exponent = torch.exp(-0.5*(torch.square(px_g[:,:,None] - x0[None,None,:])
                               +torch.square(py_g[:,:,None] - y0[None,None,:]))/sigma2)
        iwe = torch.mean(exponent,dim=-1)/(2*torch.pi*sigma2)
        iwe_mean = torch.mean(iwe)
        var = -torch.mean(torch.square(iwe-iwe_mean)) # The negative variance of IWE image
        return var, iwe
    
    
    v0 = torch.tensor([[0.0, 0.0]],device=evts.device, requires_grad=True) 
    optimizer = optim.Adam([v0], lr=1.0)  # 随机梯度下降，学习率0.1
    scheduler = CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=1e-5)
    
    for epoch in range(n_iter):
        optimizer.zero_grad()  # 清除之前的梯度
        loss,iwe = iwe_var(evts, v0, K=K, sigma2=sigma2)    # 计算目标函数值
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新参数
        scheduler.step()       # 更新学习率

        if debug:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:2d}: v = {v0.detach().cpu().numpy().flatten()}, loss = {loss.item():.3e}, lr = {lr:.6f}")
    return v0, loss