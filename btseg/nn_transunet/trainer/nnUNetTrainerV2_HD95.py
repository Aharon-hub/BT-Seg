from __future__ import annotations
import numpy as np, torch
from medpy.metric.binary import hd95 as _hd95
from .nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerV2_HD95(nnUNetTrainerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.online_eval_foreground_hd95 = []
        self.all_val_eval_hd95 = []
        self._eval_voxelspacing = None

    def _get_eval_spacing(self):
        if self._eval_voxelspacing is None:
            try:
                s = self.plans['plans_per_stage'][self.stage]['current_spacing']
                self._eval_voxelspacing = tuple(float(x) for x in s[:3])
            except Exception:
                self._eval_voxelspacing = (1.0, 1.0, 1.0)
        return self._eval_voxelspacing

    @staticmethod
    def _safe_hd95(pred_mask: np.ndarray, ref_mask: np.ndarray, spacing):
        p, r = bool(pred_mask.any()), bool(ref_mask.any())
        if not p and not r: return np.nan
        if p and r: return float(_hd95(pred_mask, ref_mask, voxelspacing=spacing))
        return np.inf

    def _hd95_regions(self, seg_pred: np.ndarray, gt: np.ndarray, spacing):
        et_p, et_r = (seg_pred == 4), (gt == 4)
        tc_p, tc_r = ((seg_pred == 1) | (seg_pred == 4)), ((gt == 1) | (gt == 4))
        wt_p, wt_r = ((seg_pred == 1) | (seg_pred == 2) | (seg_pred == 4)), ((gt == 1) | (gt == 2) | (gt == 4))
        return np.array([
            self._safe_hd95(et_p, et_r, spacing),
            self._safe_hd95(tc_p, tc_r, spacing),
            self._safe_hd95(wt_p, wt_r, spacing),
        ], dtype=np.float32)

    def run_online_evaluation(self, output, target):
        super().run_online_evaluation(output, target)
        if isinstance(output, dict): output = output.get("output", output.get("seg", output))
        if output.dim() != 5: output = output.unsqueeze(0) if output.dim()==4 else output
        pred_lbl = torch.argmax(output, dim=1).detach().cpu().numpy()
        if target.dim() == 4: target = target.unsqueeze(1)
        gt = target[:,0].detach().cpu().numpy()
        spacing = self._get_eval_spacing()
        batch_vals = [self._hd95_regions(pred_lbl[b], gt[b], spacing) for b in range(pred_lbl.shape[0])]
        self.online_eval_foreground_hd95.append(np.vstack(batch_vals))

    def on_epoch_end(self):
        cont = super().on_epoch_end()
        if len(self.online_eval_foreground_hd95):
            hd = np.vstack(self.online_eval_foreground_hd95)
            hd[np.isinf(hd)] = np.nan
            means = np.nanmean(hd, axis=0).astype(np.float32)
        else:
            means = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self.all_val_eval_hd95.append(means)
        self.print_to_log_file("Average global foreground HD95 (mm): [np.float32(%.6f), np.float32(%.6f), np.float32(%.6f)]" % (means[0], means[1], means[2]))
        self.online_eval_foreground_hd95 = []
        return cont
