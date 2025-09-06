import numpy as np
import torch
import torch.nn.functional as F


def _events_to_voxel_grid(events, num_bins, H, W, t0=None, t1=None):
    """
    Convert events (B,4) to a voxel grid tensor (C,H,W)
    events: tensor of shape [B, 4], columns: [x, y, t, p]
    num_bins: number of temporal bins (C)
    H, W: height and width of the voxel grid
    """

    assert events.shape[1] == 4
    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].float()  # polarity: -1 or 1

    x = x-torch.min(x)
    y = y-torch.min(y)

    # Normalize timestamps to [0, 1]
    if t0 is None:
        t0 = t.min()
    if t1 is None:
        t1 = t.max()
    t_norm = (t - t0) / (t1 - t0 + 1e-8)
    t_idx = (t_norm * (num_bins - 1)).clamp(0, num_bins - 1)

    # Floor and Ceil bins for linear interpolation
    t_low = t_idx.floor().long()
    t_high = t_low + 1
    w_high = t_idx - t_low.float()
    w_low = 1.0 - w_high

    # Initialize voxel grid 
    voxel_grid = torch.zeros((num_bins, H, W), dtype=torch.float32, device=events.device)

    # Accumulate using bilinear temporal weightstensor(0.2205, device='cuda:0') tensor(0.0302, device='cuda:0')
    valid_low = t_low < num_bins
    voxel_grid.index_put_((t_low[valid_low], y[valid_low], x[valid_low]),
                          w_low[valid_low], accumulate=True)

    valid_high = t_high < num_bins
    voxel_grid.index_put_((t_high[valid_high], y[valid_high], x[valid_high]), 
                          w_high[valid_high], accumulate=True)

    return voxel_grid, t1-t0

def _subpixel_correlation(imgs, deltaT, step=1, sigma=0.3, smooth=True):
    # normerlize the image
    imgs = imgs - torch.mean(imgs, dim=(1,2),keepdim=True)
    imgs = imgs/(torch.norm(imgs,dim=(1,2),keepdim=True)+1e-9)
    # perform cross correlation in FFT Frequency domain
    ffts = torch.fft.fft2(imgs)
    R = torch.conj(ffts[:-step])*ffts[step:]
    R = torch.mean(R,dim=0) # Ensemble for the I[0]*I[1],I[1]*I[2],I[2]*I[3]...

    H, W = imgs[0].shape
    if smooth:
        # # Gaussian smooth in Frequency domain
        # yy, xx = torch.meshgrid(torch.arange(H, device=imgs.device), 
        #                         torch.arange(W,device=imgs.device), indexing="ij")
        # gauss = torch.exp(-((xx - W//2)**2 + (yy - H//2)**2) / (2*sigma**2))
        # gauss = gauss / gauss.sum()
        # gauss = torch.fft.fft2(gauss)

        fy = torch.fft.fftfreq(H, d=1.0).to(imgs.device)  # shape: [H]
        fx = torch.fft.fftfreq(W, d=1.0).to(imgs.device)  # shape: [W]
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')  # [W, H]
        
        # 频域高斯函数：exp(-2π²σ²(fx² + fy²))
        gauss = torch.exp(
            -2 * (torch.pi ** 2) * sigma**2 * (FX**2 + FY**2)
            - 2j * torch.pi * (FX * (W//2) + FY * (H//2))  # 注意 W/2, H/2 的位置
        )
        
        R = R * gauss  # frequency domain smoothing
    
    # Inverse FFT to get cross-correlation
    r = torch.fft.ifft2(R).real
    
    # The subpixel with gaussian fitting
    kernel = torch.zeros([4,1,5,5])
    kernel[0, 0, 2, :] = torch.tensor([-20, 10, 20, 10,-20]) # [-20,10,20,10,-20], [0, -2, 4, 2, 0]
    kernel[1, 0, 2, :] = torch.tensor([-14, -7,  0,  7, 14]) #  [-14,-7,0,7,14], [0, -1, 0, 1, 0]
    kernel[2, 0, :, 2] = torch.tensor([-20, 10, 20, 10,-20])
    kernel[3, 0, :, 2] = torch.tensor([-14, -7,  0,  7, 14])
    kernel = kernel.to(r.device)
    r_safe = torch.clamp(r, min=1e-6)
    temp = F.conv2d(torch.log(r_safe.view(1,1,H,W)), kernel, padding=2)  # 输出形状：[1, 4, H, W]
    dx_sub = temp[0,1]/temp[0,0]
    dy_sub = temp[0,3]/temp[0,2]
    
    # Find argmax
    max_idx = torch.argmax(r)
    dy, dx = torch.div(max_idx, W, rounding_mode='floor'), max_idx % W
    loss = r[dy,dx]
    
    # Find the optimal sub-pixel 
    sx, sy = dx_sub[dy,dx], dy_sub[dy,dx]
    dx, dy = dx + sx, dy + sy
    # Fix the center difference
    dxx = dx- W // 2
    dyy = dy- H // 2
    
    # Turn the pixel/frame to pixel/ms
    coeff = len(imgs)/step/deltaT
    # print(coeff, dxx, dyy)
    dxx, dyy = coeff*dxx,coeff*dyy
    v_est = torch.Tensor([[dxx,dyy]]).to(imgs.device)
    return v_est, loss


def estimateCC(evts, num_bins=5, H=33, W=33, step=1, sigma=0.3, smooth=True):
    imgs, deltaT = _events_to_voxel_grid(evts,num_bins,H,W)
    v_est,loss = _subpixel_correlation(imgs,deltaT=deltaT,step=step,sigma=sigma,smooth=smooth)
    return v_est,loss