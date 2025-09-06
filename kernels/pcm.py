import numpy as np
import torch

# def solve(A, y):
#     """
#     Solves X = A^-1 @ y for a batch of 2x2 matrices A and vectors y,
#     but only at positions where the matrix A is negative definite
#     (i.e., A[0,0] < 0 and det(A) > 0).

#     Args:
#         A: Tensor of shape (K, 2, 2), batch of 2x2 matrices
#         y: Tensor of shape (K, 2), batch of right-hand side vectors

#     Returns:
#         res: Tensor of shape (K, 2), solution vectors where valid, zeros elsewhere
#     """
#     det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
#     mask = (A[:, 0, 0] < 0) & (det > 0)

#     # Extract valid submatrices and RHS vectors
#     a, b = A[mask, 0, 0], A[mask, 0, 1]
#     c, d = A[mask, 1, 0], A[mask, 1, 1]
#     y0, y1 = y[mask, 0], y[mask, 1]

#     # Solve using closed-form inverse of 2x2 matrix
#     delta0 = d * y0 - c * y1
#     delta1 = -b * y0 + a * y1
#     delta = torch.stack([delta0, delta1], dim=1)  # (M, 2)
#     delta = delta / det[mask][:, None]            # Normalize by determinant

#     # Fill result tensor with valid updates
#     res = torch.zeros_like(y)
#     res[mask] = delta
#     print(A.device, mask.device)
#     return res


def solve(A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fully vectorized batched solve of X = A^-1 @ y, only where A is negative definite.
    Avoids any slicing, indexing, or mask-based tensor creation.

    Args:
        A: Tensor of shape (K, 2, 2)
        y: Tensor of shape (K, 2)

    Returns:
        Tensor of shape (K, 2), valid solutions where A is negative definite, 0 elsewhere.
    """
    a, b = A[:, 0, 0], A[:, 0, 1]
    c, d = A[:, 1, 0], A[:, 1, 1]
    det = a * d - b * c

    # Mask of valid matrices (negative definite)
    mask = (a < 0) & (det > 0)

    # Use mask to create scaling factor (1/det if valid, 0 otherwise)
    inv_det = torch.where(mask, 1.0 / det, torch.zeros_like(det))

    # Construct inverse matrix entries, scale by inv_det
    invA = torch.zeros_like(A)
    invA[:, 0, 0] =  d * inv_det
    invA[:, 0, 1] = -b * inv_det
    invA[:, 1, 0] = -c * inv_det
    invA[:, 1, 1] =  a * inv_det

    # Matrix-vector multiplication for all entries
    res = torch.einsum('bij,bj->bi', invA, y)  # Shape (K, 2)

    return res


def objective2d_origin(evts, velocities, sigma2=1.0, tau2=0.0, compute_grad=True, compute_hessian=True):
    """
    Compute the PCM objective, its gradient, and Hessian.
    """
    sigma2 = sigma2 + tau2
    
    device = evts.device
    x, y, t, p = evts[:, 0:1], evts[:, 1:2], evts[:, 2:3], evts[:, 3:4]
    vx, vy = velocities[:, 0:1, None], velocities[:, 1:2, None]
    
    xij = (x.T - x)[None,:,:]  # (1, N, N)
    yij = (y.T - y)[None,:,:]  #.reshape(1, -1)
    tij = (t.T - t)[None,:,:]  #.reshape(1, -1)

    bx = xij - vx * tij  # (K, N, N)
    by = yij - vy * tij
    bx2, by2, tij2 = bx**2, by**2, tij**2
    
    exponent = torch.exp(-0.5 * (bx2 + by2) / sigma2)  # shape: (K, N²)
    coeff = 1.0 / (2 * torch.pi * sigma2)  
    kernel = coeff * exponent

    loss = torch.mean(kernel, dim=(1, 2))  # mean density over all pairs
    
    J, H = None, None
    if compute_grad:
        Jx = torch.sum(kernel*bx*tij/sigma2, dim=(1, 2)) 
        Jy = torch.sum(kernel*by*tij/sigma2, dim=(1, 2)) 
        J = torch.stack([Jx, Jy], dim=1)
    if compute_hessian:
        factor = kernel*tij2 / sigma2  # Shape (K, N, N)
        Hxx = torch.sum(factor* (bx2 / sigma2 - 1), dim=(1, 2))
        Hyy = torch.sum(factor* (by2 / sigma2 - 1), dim=(1, 2))
        Hxy = torch.sum(factor* bx * by / sigma2, dim=(1, 2))
        
        H = torch.zeros((len(velocities), 2, 2), device=device)
        H[:, 0, 0], H[:, 0, 1] = Hxx, Hxy
        H[:, 1, 0], H[:, 1, 1] = Hxy, Hyy
    return loss, J, H


# 输入: evts (N, 4), velocities (K, 2)
def objective2d_smooth(evts, velocities, sigma2=1.0, tau2=1.0, compute_grad=False, compute_hessian=False):
    device = evts.device
    evts = evts.to(torch.float32)
    velocities = velocities.to(torch.float32)

    x, y, t = evts[:, 0], evts[:, 1], evts[:, 2]
    xij = x[None, :] - x[:, None]  # (N, N)
    yij = y[None, :] - y[:, None]
    tij = t[None, :] - t[:, None]  # (N, N)

    xij = xij[None, :, :]  # (1, N, N)
    yij = yij[None, :, :]
    tij = tij[None, :, :]

    vx = velocities[:, 0].view(-1, 1, 1)
    vy = velocities[:, 1].view(-1, 1, 1)

    bx = xij - vx * tij  # (K, N, N)
    by = yij - vy * tij
    tij2 = tij ** 2
    denom = sigma2 + tij2 * tau2
    inv_denom = 1.0 / denom

    norm2 = bx ** 2 + by ** 2
    exponent = torch.exp(-0.5 * norm2 * inv_denom)
    coeff = inv_denom / (2 * 3.14159)
    kernel = coeff * exponent  # (K, N, N)

    loss = kernel.mean(dim=(1, 2))  # (K,)

    J = H = None
    if compute_grad:
        Jx = torch.einsum('kij,kij->k', kernel, bx * tij * inv_denom)
        Jy = torch.einsum('kij,kij->k', kernel, by * tij * inv_denom)
        J = torch.stack([Jx, Jy], dim=1)
    if compute_hessian:
        factor = kernel * tij2 * inv_denom
        Hxx = torch.einsum('kij,kij->k', factor, (bx ** 2 * inv_denom - 1))
        Hyy = torch.einsum('kij,kij->k', factor, (by ** 2 * inv_denom - 1))
        Hxy = torch.einsum('kij,kij->k', factor, bx * by * inv_denom)
        H = torch.zeros((len(velocities), 2, 2), device=device)
        H[:, 0, 0], H[:, 1, 1] = Hxx, Hyy
        H[:, 0, 1] = H[:, 1, 0] = Hxy
    return loss, J, H



def estimatePCM(evts, objective2d=objective2d_smooth, sigma2_min=1.0, iter_n=12):
    """
    Estimate the velocity that maximizes the event-based objective function using second-order optimization.

    Parameters:
    ----------
    evts : torch.Tensor
        Input event data with shape (N, 4), where each row is (x, y, t, p).
    
    objective2d : callable
        Objective function that returns loss, gradient (Jacobian), and Hessian given the current velocity estimate.
        Default is the smoothed version: objective2d_smooth.
    
    iter_n : int
        Number of iterations for the Newton optimization loop.

    Method:
    ------
    This function performs Newton iterations to minimize a 2D objective function.
    Both the Gaussian kernel variance (sigma²) and the convolution smoothing kernel variance (tau²)
    are exponentially decayed over iterations to progressively refine the solution.

    Returns:
    -------
    vn : torch.Tensor
        Final estimated velocity as a (1, 2) tensor.
    
    loss : torch.Tensor
        Final objective loss value.
    """
    assert objective2d in [objective2d_smooth, objective2d_origin],"Error"
    
    decay1 = (iter_n - 1) / (np.log(2.0) - np.log(sigma2_min))+1e-9 # exponential decay
    sigma2_list = np.array([2.0 * np.exp(-k / decay1) for k in range(iter_n)])
    
    decay2 = (iter_n - 1) / (np.log(1e4) - np.log(1e-4))+1e-9 # exponential decay
    tau2_list = np.array([1e4 * np.exp(-k / decay2) for k in range(iter_n)])

    vn = torch.zeros([1,2], device=evts[0].device)
    for sigma2, tau2 in zip(sigma2_list, tau2_list):
        if isinstance(evts, list):
            # Multiple event blocks: accumulate loss, gradient, and Hessian
            total_loss = 0.0
            total_J = torch.zeros_like(vn)
            total_H = torch.zeros((1, 2, 2))
            for e in evts:
                loss, J, H = objective2d(e, vn, sigma2=sigma2, tau2=tau2, compute_grad=True, compute_hessian=True)
                total_loss += loss
                total_J += J
                total_H += H
            loss, J, H = total_loss, total_J, total_H
        else:
            # Single event block
            loss, J, H = objective2d(evts, vn, sigma2=sigma2, tau2=tau2, compute_grad=True, compute_hessian=True)

        delta = solve(H, J)
        delta = torch.clamp(delta, min=-5, max=5)
        vn = vn - delta
    return vn, loss


def estimatePCMB(evts, sigma2_min=1.0, iter_n=64, **kwargs):
    return estimatePCM(evts, objective2d=objective2d_origin, sigma2_min=sigma2_min, iter_n=iter_n)


def estimatePCMSA(evts, sigma2_min=1.0, iter_n=12, **kwargs):
    return estimatePCM(evts, objective2d=objective2d_smooth, sigma2_min=sigma2_min, iter_n=iter_n)