# import torch
# import torch.nn as nn
# import auraloss


# class MultiResoFuseLoss(nn.Module):
#     def __init__(self, l1_ratio=0, **kwargs) -> None:
#         super().__init__()

#         self.l1_ratio = l1_ratio
#         self.l1 = nn.L1Loss()
#         self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(**kwargs)

#     def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
#         """
#         est: (B, C, T)
#         gt: (B, C, T)
#         """
#         B, C, T = est.shape

#         if self.l1_ratio > 0:
#             loss1 = self.loss_fn(est, gt) + self.l1_ratio * self.l1(est, gt)
#         else:
#             loss1 = self.loss_fn(est, gt)

#         return loss1

# AFTER
import torch
import torch.nn as nn
import auraloss
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as si_sdr_fn

class MultiResoFuseLoss(nn.Module):
    def __init__(self, l1_ratio=0, si_sdr_ratio=0.0, **kwargs) -> None:
        super().__init__()
        self.l1_ratio = l1_ratio
        self.si_sdr_ratio = si_sdr_ratio
        self.l1 = nn.L1Loss()
        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(**kwargs)

    def forward(self, est, gt, **kwargs):
        B, C, T = est.shape
        freq_loss = self.loss_fn(est, gt)
        if self.l1_ratio > 0:
            freq_loss = freq_loss + self.l1_ratio * self.l1(est, gt)
        if self.si_sdr_ratio > 0:
            est_flat = est.reshape(B * C, T)
            gt_flat = gt.reshape(B * C, T)
            si_sdr_val = si_sdr_fn(est_flat, gt_flat).mean()
            freq_loss = freq_loss - self.si_sdr_ratio * si_sdr_val
        return freq_loss
