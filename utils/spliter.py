# import torch
from collections import defaultdict

import torch
from addict import Dict

def filter_events_by_time_polarity(evts, t_min, t_max, polarity=None):
    # evts: [B,4]  tensor
    t = evts[:, 2]
    p = evts[:, 3]

    mask_t = (t >= t_min) & (t < t_max)
    if polarity is None:
        mask = mask_t
    else:
        mask_p = (p == polarity)
        mask = mask_t & mask_p

    filtered_evts = evts[mask]
    return filtered_evts


class EventSpatialSplitter():
    def __init__(self, cfg):
        self._c = cfg
        x_span, y_span = self._c.x_span, self._c.y_span
        
        # 使用整数坐标范围
        tx = torch.arange(-x_span//2+1, x_span//2+1, device='cpu')  # int64 by default
        ty = torch.arange(-y_span//2+1, y_span//2+1, device='cpu')
        
        txx, tyy = torch.meshgrid(tx, ty, indexing="ij")
        self.targets = torch.stack([txx.flatten(), tyy.flatten()], dim=1).to(torch.int64)  # [B,2]
    
    def put_data(self, evts):
        self.evts = evts
        x = evts[:, 0].to(torch.int64)
        y = evts[:, 1].to(torch.int64)

        pixel_keys = x + y * 10000  # 假设 x < 10000
        uniq_pixel_keys, pixel_inv = torch.unique(pixel_keys, return_inverse=True)
        self.uniq_pixel_keys = uniq_pixel_keys
        self.pixel_inv = pixel_inv
        self.evts_device = evts.device

    def obtain_at(self, x_shift, y_shift):
        # x_shift, y_shift: 标量或张量，用于偏移targets
        # 先移动targets
        shifted_targets = self.targets + torch.tensor([x_shift, y_shift], device=self.targets.device)
        target_keys = shifted_targets[:, 0] + shifted_targets[:, 1] * 10000

        # 判断uniq_pixel_keys哪些匹配
        mask = torch.isin(self.uniq_pixel_keys.to(target_keys.device), target_keys)

        idx = torch.nonzero(mask, as_tuple=False).flatten()
        # print("匹配的 uniq_pixels 索引:", idx)

        mask_evts = torch.isin(self.pixel_inv, idx.to(self.pixel_inv.device))
        evts_at = self.evts[mask_evts]
        return evts_at

    def obtain_all(self):
        xs_pos = torch.arange(0, self._c.H, self._c.step, device='cpu')
        ys_pos = torch.arange(0, self._c.W, self._c.step, device='cpu')
        
        blocks = [[None for _ in ys_pos] for _ in xs_pos]
        centers = [[None for _ in ys_pos] for _ in xs_pos]
        for i, x0 in enumerate(xs_pos):
            for j, y0 in enumerate(ys_pos):
                blocks[i][j] = self.obtain_at(x0,y0)
                centers[i][j] = (x0, y0)
        return blocks, centers


def test():
    cfg = Dict()
    cfg.x_span = 32
    cfg.y_span = 32
    cfg.H, cfg.W = 256, 256
    cfg.step = 32
    cfg.t_min = 0
    cfg.t_max = 256
    
    spliter = EventSpatialSplitter(cfg)
    
    # 模拟事件
    torch.manual_seed(42)
    evts_all = torch.randint(0, 256, (100000000, 4)).float().to("cuda:0")

    evts = filter_events_by_time_polarity(evts_all, t_min=cfg.t_min, t_max=cfg.t_max, polarity=1)
    spliter.put_data(evts)
    print(evts)
    
    # 查询偏移位置为 (2, 2) 的事件子集
    result = spliter.obtain_at(32, 32)
    print(result)

    print(evts.device, evts_all.device, result.device)
    
    
    # blocks, centers = spliter.obtain_all()

if __name__ == '__main__':
    test()


 