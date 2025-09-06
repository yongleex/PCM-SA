import torch
import numpy as np
from addict import Dict

from utils.spliter import filter_events_by_time_polarity, EventSpatialSplitter

from kernels.correlation import estimateCC
from kernels.opticalflow import estimateOF
from kernels.cmax import estimateCMAX
from kernels.pcm import estimatePCMB, estimatePCMSA


class PIVFramework:
    def __init__(self, cfg):
        self._c = cfg

        if self._c.method not in [estimateCC, estimateOF, estimateCMAX, estimatePCMB,estimatePCMSA]:
            raise ValueError("Please provide method in cfg.method")

        # 初始化分块器
        self.spliter = EventSpatialSplitter(self._c)

    def compute(self, events, polarity=1):
        print("Start splitting event blocks...")
        evts = filter_events_by_time_polarity(events, t_min=self._c.t_min, t_max=self._c.t_max, polarity=polarity)
        # evts[:,2] = evts[:,2] - torch.mean(evts[:,2]) 
        self.spliter.put_data(evts)

        cx = torch.arange(0, self._c.H+1, self._c.step)+self._c.start[0]
        cy = torch.arange(0, self._c.W+1, self._c.step)+self._c.start[1]
        xx,yy = torch.meshgrid(cx,cy,indexing='ij')
        
        print("Start velocity estimation per block...", evts.device)
        u, v = np.zeros((len(cx),len(cy))), np.zeros((len(cx),len(cy)))
        for i,px in enumerate(cx):
            for j,py in enumerate(cy):
                block = self.spliter.obtain_at(px, py)[:self._c.max_events] # 这是个点
                if block is None or len(block) < self._c.min_events:
                    u[i,j], v[i,j] = np.nan, np.nan
                    continue
                v_est, loss = self._c.method(
                    block, **self._c.method_args)
                v_est = v_est.cpu().detach().numpy()
                u[i,j] = v_est[0,0]
                v[i,j] = v_est[0,1]
        return u, v, (xx,yy)