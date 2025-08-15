from __future__ import annotations
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from ..trainer.nnUNetTrainer import nnUNetTrainer
from ..trainer.loss_functions import MultipleOutputLoss2
from ..trainer.network_trainer import maybe_to_torch, to_cuda
from ..data.default_data_augmentation import default_2D_augmentation_params, default_3D_augmentation_params, get_patch_size
from ..data.data_augmentation_moreDA import get_moreDA_augmentation
from ..data.dataset_loading import unpack_dataset
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
from typing import Tuple, Dict, Any
from inspect import signature

softmax_helper = lambda x: F.softmax(x, 1)

def poly_lr(epoch: int, max_epochs: int, initial_lr: float, exponent: float = 0.9):
    return initial_lr * (1 - epoch / float(max_epochs)) ** exponent

class InitWeights_He:
    def __init__(self, neg_slope: float = 1e-2): self.neg_slope = neg_slope
    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None: module.bias = nn.init.constant_(module.bias, 0.0)

class nnUNetTrainerV2(nnUNetTrainer):
    def __init__(self, plans_file: str, fold: int | str, output_folder: str | None = None, dataset_directory: str | None = None, batch_dice: bool = True, stage: int | None = None, unpack_data: bool = True, deterministic: bool = True, fp16: bool = False, input_size: Tuple[int,int,int]=(64,160,160), args: Any | None = None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16)
        self.model_params: Dict[str, Any] = getattr(args, "model_params", {})
        self.input_size = input_size
        self.model: str = args.model if args else "Generic_TransUNet_max_ppbp"
        self.resume: str = args.resume if args else ""
        self.disable_ds: bool = args.disable_ds if args else False
        self.max_num_epochs: int = args.max_num_epochs if args else 1000
        self.initial_lr: float = args.initial_lr if args else 1e-3
        self.args = args
        self.save_every = 1
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1,1,1]] + list(list(i) for i in 1/np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params["rotation_x"] = (-30/360*2*np.pi, 30/360*2*np.pi)
            self.data_aug_params["rotation_y"] = (-30/360*2*np.pi, 30/360*2*np.pi)
            self.data_aug_params["rotation_z"] = (-30/360*2*np.pi, 30/360*2*np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                for k in ("elastic_deform_alpha","elastic_deform_sigma","rotation_x"):
                    self.data_aug_params[k] = default_2D_augmentation_params[k]
            else:
                self.do_dummy_2D_aug = False
            if max(self.patch_size)/min(self.patch_size) > 1.5:
                default_2D_augmentation_params["rotation_x"] = (-15/360*2*np.pi, 15/360*2*np.pi)
                self.data_aug_params = default_2D_augmentation_params
            self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
            if self.do_dummy_2D_aug:
                self.basic_generator_patch_size = get_patch_size(self.patch_size[1:], self.data_aug_params["rotation_x"], self.data_aug_params["rotation_y"], self.data_aug_params["rotation_z"], self.data_aug_params["scale_range"])
                self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            else:
                self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params["rotation_x"], self.data_aug_params["rotation_y"], self.data_aug_params["rotation_z"], self.data_aug_params["scale_range"])
            self.data_aug_params["scale_range"] = (0.7, 1.4)
            self.data_aug_params["do_elastic"] = False
            self.data_aug_params["selected_seg_channels"] = [0]
            self.data_aug_params["patch_size_for_spatialtransform"] = self.patch_size
            self.data_aug_params["num_cached_per_thread"] = 1
        else:
            self.data_aug_params = default_2D_augmentation_params

    def initialize(self, training: bool = True, force_load_plans: bool = False):
        if self.was_initialized: return
        maybe_mkdir_p(self.output_folder)
        if force_load_plans or self.plans is None: self.load_plans_file()
        self.process_plans(self.plans); self.setup_DA_params()
        net_numpool = len(self.net_num_pool_op_kernel_sizes)
        weights = np.array([1/(2**i) for i in range(net_numpool)])
        mask = np.array([True]+[True if i<net_numpool-1 else False for i in range(1,net_numpool)])
        weights[~mask]=0; weights/=weights.sum(); self.ds_loss_weights = weights
        if self.disable_ds:
            from ..trainer.loss_functions import DC_and_CE_loss
            self.loss = DC_and_CE_loss({"batch_dice": self.batch_dice, "smooth": 1e-5, "do_bg": False}, {})
        else:
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
        self.folder_with_preprocessed_data = join(self.dataset_directory, f"{self.plans['data_identifier']}_stage{self.stage}")
        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                unpack_dataset(self.folder_with_preprocessed_data)
            _gmda_kwargs = dict(deep_supervision_scales=self.deep_supervision_scales, pin_memory=self.pin_memory, use_nondetMultiThreadedAugmenter=False)
            sig = signature(get_moreDA_augmentation)
            if "num_threads_in_multithreaded" in sig.parameters: _gmda_kwargs["num_threads_in_multithreaded"]=2
            elif "num_processes" in sig.parameters: _gmda_kwargs["num_processes"]=2
            self.tr_gen, self.val_gen = get_moreDA_augmentation(self.dl_tr, self.dl_val, self.data_aug_params["patch_size_for_spatialtransform"], self.data_aug_params, **_gmda_kwargs)
        self.initialize_network(); self.initialize_optimizer_and_scheduler(); self.was_initialized=True

    def initialize_network(self):
        if self.threeD: conv_op=nn.Conv3d; dropout_op=nn.Dropout3d; norm_op=nn.InstanceNorm3d
        else: conv_op=nn.Conv2d; dropout_op=nn.Dropout2d; norm_op=nn.InstanceNorm2d
        norm_op_kwargs={"eps":1e-5,"affine":True}; dropout_op_kwargs={"p":0.0,"inplace":True}
        net_nonlin=nn.LeakyReLU; net_nonlin_kwargs={"negative_slope":1e-2,"inplace":True}
        do_ds = not self.disable_ds
        available_keys = list(self.plans["plans_per_stage"].keys())
        resolution_index = 1 if 1 in available_keys else available_keys[0]
        patch_size_from_plans = self.plans["plans_per_stage"][resolution_index]["patch_size"]
        model_params = dict(self.model_params)
        for key in ("patch_size","img_size"):
            if key in model_params: model_params.pop(key)
        if self.model == "Generic_TransUNet_max_ppbp":
            from ..networks.transunet3d_model import Generic_TransUNet_max_ppbp
            self.network = Generic_TransUNet_max_ppbp(self.num_input_channels, self.base_num_features, self.num_classes, len(self.net_num_pool_op_kernel_sizes), self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, do_ds, False, lambda x: x, InitWeights_He(1e-2), self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, convolutional_upsampling=(False if ("is_fam" in model_params and model_params["is_fam"]) else True), patch_size=patch_size_from_plans, **model_params)
        else:
            raise NotImplementedError(self.model)
        if torch.cuda.is_available(): self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        if getattr(self, "lrschedule", "poly") == "cosine": 
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, betas=(0.9,0.999))
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_num_epochs, eta_min=1e-6)
        else:
            super().initialize_optimizer_and_scheduler()

    def maybe_update_lr(self, epoch: int | None = None):
        ep = self.epoch + 1 if epoch is None else epoch
        self.optimizer.param_groups[0]["lr"] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)

    def run_iteration(self, data_generator, do_backprop: bool = True, run_online_evaluation: bool = False):
        data_dict = next(data_generator); data = data_dict["data"]; target = data_dict["target"]
        data = maybe_to_torch(data); target = maybe_to_torch(target)
        if torch.cuda.is_available(): data = to_cuda(data); target = to_cuda(target)
        self.optimizer.zero_grad()
        if self.fp16:
            with autocast():
                output = self.network(data); del data
                output_for_loss = output if not isinstance(output, dict) else output.get("output", output.get("seg", output))
                if isinstance(output, (list, tuple)): output_for_loss = output if not self.disable_ds else output[0]
                target_for_loss = target if not isinstance(target, (list, tuple)) else (target if not self.disable_ds else target[0])
                l = self.loss(output_for_loss, target_for_loss)
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer); self.amp_grad_scaler.update()
        else:
            output = self.network(data); del data
            output_for_loss = output if not isinstance(output, dict) else output.get("output", output.get("seg", output))
            if isinstance(output, (list, tuple)): output_for_loss = output if not self.disable_ds else output[0]
            target_for_loss = target if not isinstance(target, (list, tuple)) else (target if not self.disable_ds else target[0])
            l = self.loss(output_for_loss, target_for_loss)
            if do_backprop:
                l.backward(); torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12); self.optimizer.step()
        loss_val = l.detach().cpu().numpy()
        if run_online_evaluation:
            out_eval = output[0] if isinstance(output, (list,tuple)) else output
            tgt_eval = target[0] if isinstance(target, (list,tuple)) else target
            if isinstance(out_eval, dict): out_eval = out_eval.get("output", out_eval.get("seg", out_eval))
            if out_eval.dim() != 5: out_eval = out_eval.unsqueeze(0)
            if tgt_eval.dim() == 4: tgt_eval = tgt_eval.unsqueeze(1)
            self.run_online_evaluation(out_eval, tgt_eval)
        del target
        return loss_val
