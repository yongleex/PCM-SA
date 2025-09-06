import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

# from skimage.registration import optical_flow_ilk
import cv2


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

def optical_flow(imgs, deltaT, step=1, sigma=0.3, smooth=True):
    # normerlize the image
    imgs = imgs.cpu().detach().numpy()
    # imgs = (imgs*100/np.mean(imgs)).astype(np.uint8)
    v_list = list()
    N, H, W = imgs.shape 
    for i in range(N-step):
        img1, img2 = imgs[i], imgs[i+step]
        img1, img2 = cv2.blur(img1, (7,7)), cv2.blur(img2, (7,7))
        img1 = (img1*255/np.max(img1)).astype(np.uint8)
        img2 = (img2*255/np.max(img2)).astype(np.uint8)
        p0 = np.array([[W//2, H//2]], dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, winSize=(25,25),maxLevel=4)
        v_list.append(p1-p0)

    u = np.mean(np.array(v_list)[:,0,0,:],axis=0)
    coeff = len(imgs)/step/deltaT
    dxx = u*coeff.cpu().detach().numpy()
    v_est = torch.Tensor(np.array([dxx]))
    return v_est, None


def estimateOF(evts, num_bins=5, H=33, W=33, step=1, sigma=0.3, smooth=True):
    imgs, deltaT = _events_to_voxel_grid(evts,num_bins,H,W)
    v_est,loss = optical_flow(imgs,deltaT=deltaT,step=step,sigma=sigma,smooth=smooth)
    v_est = v_est.to(evts.device)
    return v_est,loss