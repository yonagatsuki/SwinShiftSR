import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext

from datapipe.datasets import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt


import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

# Import SwinIR
from stage1_utils.swinir import SwinIR


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs
        self.setup_dist()
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )
        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}
        
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))
            
        self.last_log_time = time.time()

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                ckpt_key = key
                if key not in ckpt and 'module.' + key in ckpt:
                    ckpt_key = 'module.' + key
                
                if ckpt_key in ckpt:
                    ema_state[key] = deepcopy(ckpt[ckpt_key].detach().data)

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            
            state_dict = ckpt['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v 
                else:
                    new_state_dict[k] = v
            

            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model

            util_net.reload_model(model_to_load, new_state_dict)

            
            torch.cuda.empty_cache()

            self.iters_start = ckpt['iters_start']
            

            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                if ema_ckpt_path.exists():
                    ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                    _load_ema_state(self.ema_state, ema_ckpt)
                else:
                    self.logger.warning(f"EMA Checkpoint not found at {ema_ckpt_path}")
            
            torch.cuda.empty_cache()

            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False) 
        else:
            self.model = model

        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2),
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        if isinstance(data, dict):
             data = {
                key: value.cuda().to(dtype=dtype) if isinstance(value, torch.Tensor) else value 
                for key, value in data.items()
            }
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()
        self.build_model()
        self.setup_optimizaton()
        self.resume_from_ckpt()
        self.build_dataloader()

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        
        self.last_log_time = time.time()
        
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1
            data = self.prepare_data(next(self.dataloaders['train']))
            self.training_step(data)
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()
               

            if self.configs.train.use_amp:

                self.amp_scaler.unscale_(self.optimizer) 
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            
            
            self.adjust_lr()    
            self.model.zero_grad()    

            
            if hasattr(self.configs.train, 'ema_rate'):
                 self.update_ema_model()
                 
                 
            if (ii+1) % self.configs.train.save_freq == 0:
                self.save_ckpt()
            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            ckpt = {
                    'iters_start': self.current_iters,
                    'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                    'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                    'state_dict': self.model.state_dict(),
                    }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def logging_image(self, im_tensor, tag, phase, idx=0, add_global_step=False, nrow=8):
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True)
        
        if self.local_logging:
            # [Modified] Filename format: {step}_{batch_idx}_{tag}.png
            # Example: 005000_001_lq.png (Step 5000, Batch 1)
            im_path = str(self.image_dir / phase / f"{self.current_iters:06d}_{idx:03d}_{tag}.png")
            
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
            
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1
            
    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        if self.tf_logging:
            prefix = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor) and value.numel() > 1:
                        scalar_value = value.mean().item()
                    elif isinstance(value, torch.Tensor):
                        scalar_value = value.item()
                    else:
                        scalar_value = value
                    self.writer.add_scalar(f"{prefix}/{key}", scalar_value, self.log_step[phase])
            else:
                if isinstance(metrics, torch.Tensor) and metrics.numel() > 1:
                    scalar_value = metrics.mean().item()
                elif isinstance(metrics, torch.Tensor):
                    scalar_value = metrics.item()
                else:
                    scalar_value = metrics
                self.writer.add_scalar(prefix, scalar_value, self.log_step[phase])

            if add_global_step:
                self.log_step[phase] += 1

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')

class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )

    pass

class TrainerDifIRLPIPS(TrainerDifIR):
 
    def build_model(self):
       
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)
        else:
            self.model = model

        # EMA
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        self.print_model_info()

  
        if 'swinir' in self.configs:
            if self.rank == 0:
                self.logger.info(f"Loading Stage 1 SwinIR from {self.configs.swinir.ckpt_path}...")
            swinir_params = self.configs.swinir.params
            self.swinir = SwinIR(**swinir_params).cuda()
            ckpt = torch.load(self.configs.swinir.ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
            new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            self.swinir.load_state_dict(new_ckpt, strict=True)
            self.swinir.eval()
            for param in self.swinir.parameters():
                param.requires_grad = False

        # Autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            if self.configs.autoencoder.tune_decoder:
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                if self.rank == 0:
                    num_params = 0
                    for key, value in autoencoder.named_parameters():
                        if 'decoder' in key or 'post_quant_conv' in key:
                            num_params += value.numel()
                        else:
                            value.requires_grad = False
                    self.logger.info(f'Finetuning Decoder module: {num_params/10**6:.2f}M...')
            else:
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                self.freeze_model(autoencoder)
                autoencoder.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        if self.configs.autoencoder.params.lora_tune_decoder or self.configs.autoencoder.tune_decoder:
            self.freeze_model(self.model)

        # LPIPS metric
        if hasattr(self.configs, 'lpips'):
            lpips_net = self.configs.lpips.net
        else:
            lpips_net = 'vgg'
        if self.rank == 0:
            self.logger.info(f"Loading LIIPS Metric: {lpips_net}...")
        lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}")
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)
        
        self.current_x0_pred = None

    def adjust_lr(self, current_iters=None):
       
        curr_iter = current_iters if current_iters is not None else self.current_iters
        
        warmup_iters = self.configs.train.warmup_iterations
        
        base_lr = self.configs.train.lr 
        
       
        if curr_iter < warmup_iters:
            alpha = float(curr_iter) / float(max(1, warmup_iters))
            warmup_lr = base_lr * alpha
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        

        else:

            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                self.lr_scheduler.step()
            else:

                pass


    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                    )
            self.current_x0_pred = x0_pred.detach()

            losses["lpips"] = self.lpips_loss(
                    x0_pred,
                    micro_data['gt'],
                    ).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)
            losses["lpips"] *= loss_coef[1]

            if loss_coef[0] > 0:
                losses["mse"] *= loss_coef[0]
            else:
                assert loss_coef[2] > 0
                losses["mse"] = mean_flat((x0_pred - micro_data['gt']) ** 2)
                losses["mse"] *= loss_coef[2]

            assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    # [Modified] Training Step with Online Pipeline
    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            
            # =========================================================
            # Online Pipeline: SwinIR -> Flip -> Resshift
            # =========================================================
            with torch.no_grad():
                lq_raw = micro_data['lq']
                lq_01 = (lq_raw + 1.0) / 2.0
                
                lq_input_swinir = F.interpolate(lq_01, size=(128, 128), mode='bilinear', align_corners=False)
                # Save raw input for logging
                micro_data['lq_raw'] = lq_raw.clone()

                lq_output_swinir = self.swinir(lq_input_swinir)
                
                lq_output_swinir = torch.clamp(lq_output_swinir, 0.0, 1.0)
                
                
                
                gt_512 = micro_data['gt']
                
                if random.random() < 0.5:
                    lq_output_swinir = torch.flip(lq_output_swinir, dims=[3])
                    gt_512 = torch.flip(gt_512, dims=[3])

                    micro_data['lq_raw'] = torch.flip(micro_data['lq_raw'], dims=[3])
                    
                micro_data['lq'] = (lq_output_swinir - 0.5) * 2
                micro_data['gt'] = gt_512


            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
                    )
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            if 'autoencoder' in self.configs:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                    )
            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq':micro_data['lq'],}
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                model_kwargs = None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach())


    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if not hasattr(self, 'loss_mean') or self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                self.last_log_time = time.time()

            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                
                current_time = time.time()
                if not hasattr(self, 'last_log_time'): self.last_log_time = current_time - 1.0
                time_cost = current_time - self.last_log_time
                step_time = time_cost / self.configs.train.log_freq[0]
                remaining_steps = self.configs.train.iterations - self.current_iters
                eta_seconds = remaining_steps * step_time
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                self.last_log_time = current_time

                log_str = 'Train: {:06d}/{:06d}, ETA: {}, MSE/LPIPS: '.format(
                        self.current_iters,
                        self.configs.train.iterations,
                        eta_str)

                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['mse'][jj].item(),
                            self.loss_mean['lpips'][jj].item(),
                            )
                log_str += 'lr:{:.2e}, {:.2f}s/it'.format(self.optimizer.param_groups[0]['lr'], step_time)
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.log_freq[1] == 0:
                if 'lq_raw' in batch:

                    self.logging_image(batch['lq_raw'], tag='lq_noisy_raw', phase=phase, idx=0, add_global_step=False)
                if 'gt' in batch:
                    self.logging_image(batch['gt'], tag='gt', phase=phase, idx=0, add_global_step=False)
                if 'lq' in batch:
                    self.logging_image(batch['lq'], tag='lq_swinir_clean', phase=phase, idx=0, add_global_step=False)
                
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, idx=0, add_global_step=False)
                
                if self.current_x0_pred is not None:
                    self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, idx=0, add_global_step=True)
                    
            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)
    

    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            val_loss_meter = {'mse': 0.0, 'lpips': 0.0, 'total': 0.0}
            val_batch_count = 0
            loss_coef = self.configs.train.get('loss_coef', [1.0, 1.0, 0.0])
            

            has_gt = False 

            indices = np.linspace(0, self.base_diffusion.num_timesteps, self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4, endpoint=False, dtype=np.int64).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            
          
            val_seed = self.configs.train.get('seed', 12345)
            rng_gen = torch.Generator(device=f"cuda:{self.rank}")
            rng_gen.manual_seed(val_seed)

           
            rng_state_torch = torch.get_rng_state()
            rng_state_cuda = torch.cuda.get_rng_state_all()
            rng_state_numpy = np.random.get_state()
            rng_state_random = random.getstate()

            
            torch.manual_seed(val_seed)
            torch.cuda.manual_seed_all(val_seed)
            np.random.seed(val_seed)
            random.seed(val_seed)

            with torch.no_grad():
                for ii, data in enumerate(self.dataloaders[phase]):
                    data = self.prepare_data(data, phase='val')
                    
                    
                    im_gt = None 
                    
                    if 'gt' in data:
                        has_gt = True
                        im_lq_raw, im_gt = data['lq'], data['gt']
                    else:
                        im_lq_raw = data['lq']


                    im_lq_clean = im_lq_raw
                    if hasattr(self, 'swinir'):
                      
                        im_lq_01 = (im_lq_raw + 1.0) / 2.0
                        
                        im_lq_input = F.interpolate(im_lq_01, size=(128, 128), mode='bilinear', align_corners=False)
                        

                        im_lq_clean_128 = self.swinir(im_lq_input)
                        im_lq_clean_128 = torch.clamp(im_lq_clean_128, 0.0, 1.0)
                        

                        im_lq_clean = (im_lq_clean_128 - 0.5) * 2
                    
                    im_lq_final = im_lq_clean

                    if 'gt' in data:

                        t_val = torch.randint(
                            0, self.base_diffusion.num_timesteps,
                            size=(im_gt.shape[0],),
                            device=im_gt.device,
                            generator=rng_gen 
                        )
                        
                        if self.configs.model.params.cond_lq:
                            model_kwargs_val = {'lq': im_lq_final}
                            if 'mask' in data: model_kwargs_val['mask'] = data['mask']
                        else:
                            model_kwargs_val = None

                        latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
                        latent_resolution = im_gt.shape[-1] // latent_downsamping_sf
                        if 'autoencoder' in self.configs:
                            noise_chn = self.configs.autoencoder.params.embed_dim
                        else:
                            noise_chn = im_gt.shape[1]
                        

                        noise_val = torch.randn(
                            size=(im_gt.shape[0], noise_chn,) + (latent_resolution, ) * 2,
                            device=im_gt.device,
                            generator=rng_gen 
                        )

                        model_to_use = self.ema_model if self.configs.train.use_ema_val else self.model
                        
                        val_losses_out, _, z0_pred_val = self.base_diffusion.training_losses(
                            model_to_use, 
                            im_gt, 
                            im_lq_final, 
                            t_val, 
                            first_stage_model=self.autoencoder,
                            model_kwargs=model_kwargs_val,
                            noise=noise_val
                        )

                        x0_pred_val = self.base_diffusion.decode_first_stage(
                            z0_pred_val,
                            self.autoencoder,
                        )

                        mse_val = mean_flat((x0_pred_val - im_gt) ** 2)
                        lpips_val = self.lpips_loss(x0_pred_val, im_gt).view(-1)
                        if torch.any(torch.isnan(lpips_val)):
                            lpips_val = torch.nan_to_num(lpips_val, nan=0.0)

                        curr_mse_loss = mse_val.mean()
                        curr_lpips_loss = lpips_val.mean()
                        
                        weighted_lpips = curr_lpips_loss * loss_coef[1]
                        if loss_coef[0] > 0:
                            weighted_mse = curr_mse_loss * loss_coef[0]
                        else:
                            weighted_mse = curr_mse_loss * loss_coef[2]
                        
                        total_loss_val = weighted_mse + weighted_lpips

                        val_loss_meter['mse'] += weighted_mse.item()
                        val_loss_meter['lpips'] += weighted_lpips.item()
                        val_loss_meter['total'] += total_loss_val.item()
                        val_batch_count += 1

    
                    num_iters = 0
                    if self.configs.model.params.cond_lq:
                        model_kwargs = {'lq':im_lq_final,}
                        if 'mask' in data:
                            model_kwargs['mask'] = data['mask']
                    else:
                        model_kwargs = None
                    
                    tt = torch.tensor([self.base_diffusion.num_timesteps, ]*im_lq_final.shape[0], dtype=torch.int64).cuda()
                    
  
                    
                    if im_gt is not None:
    
                         hr_h, hr_w = im_gt.shape[2], im_gt.shape[3]
                    else:
          
                         hr_h, hr_w = im_lq_final.shape[2] * 4, im_lq_final.shape[3] * 4

                    if 'autoencoder' in self.configs:
        
                        h_latent = hr_h // latent_downsamping_sf
                        w_latent = hr_w // latent_downsamping_sf
                        c_latent = self.configs.autoencoder.params.embed_dim
                        noise_shape = (im_lq_final.shape[0], c_latent, h_latent, w_latent)
                    else:
                 
                        noise_shape = (im_lq_final.shape[0], im_lq_final.shape[1], hr_h, hr_w)
                    
                    fixed_start_noise = torch.randn(
                        noise_shape, 
                        device=f"cuda:{self.rank}",
                        generator=rng_gen # Use fixed generator
                    )

                    for sample in self.base_diffusion.p_sample_loop_progressive(
                            y=im_lq_final,
                            model=self.ema_model if self.configs.train.use_ema_val else self.model,
                            first_stage_model=self.autoencoder,
                            noise=fixed_start_noise, # Pass the HR fixed noise
                            clip_denoised=True if self.autoencoder is None else False,
                            model_kwargs=model_kwargs,
                            device=f"cuda:{self.rank}",
                            progress=False,
                            ):
                        sample_decode = {}
                        
                        is_visualization_step = (num_iters in indices)
                        is_last_step = (num_iters == self.base_diffusion.num_timesteps - 1)

                        if is_visualization_step or (is_last_step and 'gt' in data):
                            for key, value in sample.items():
                                if key in ['sample', ]:
                                    sample_decode[key] = self.base_diffusion.decode_first_stage(
                                            value,
                                            self.autoencoder,
                                            ).clamp(-1.0, 1.0)
                                    
                                    if is_visualization_step:
                                        im_sr_progress = sample_decode['sample']
                                        if num_iters == indices[0]: 
                                            im_sr_all = im_sr_progress
                                        else:
                                            if 'im_sr_all' in locals():
                                                im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                                            else:
                                                im_sr_all = im_sr_progress

                                    if is_last_step and 'gt' in data:

                                        mean_psnr += util_image.batch_PSNR(
                                                sample_decode['sample'] * 0.5 + 0.5,
                                                im_gt * 0.5 + 0.5,
                                                ycbcr=self.configs.train.val_y_channel,
                                                )
                                        mean_lpips += self.lpips_loss(
                                                sample_decode['sample'],
                                                im_gt,
                                                ).sum().item()

                        num_iters += 1
                        tt -= 1

                    if (ii + 1) % self.configs.train.log_freq[2] == 0:
                        self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')
                        if 'im_sr_all' in locals():
                            im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq_final.shape[1])
                            self.logging_image(im_sr_all, tag='progress', phase=phase, idx=ii, add_global_step=False, nrow=len(indices))
                        
                        if 'gt' in data:
                            self.logging_image(im_gt, tag='gt', phase=phase, idx=ii, add_global_step=False)
                        
                        im_lq_raw_log = (im_lq_raw - 0.5) * 2
                        self.logging_image(im_lq_raw_log, tag='lq_raw', phase=phase, idx=ii, add_global_step=False)
                        self.logging_image(im_lq_clean, tag='lq_clean_swinir', phase=phase, idx=ii, add_global_step=False)


            torch.set_rng_state(rng_state_torch)
            torch.cuda.set_rng_state_all(rng_state_cuda)
            np.random.set_state(rng_state_numpy)
            random.setstate(rng_state_random)


            if has_gt:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                
                if val_batch_count > 0:
                    val_loss_meter['mse'] /= val_batch_count
                    val_loss_meter['lpips'] /= val_batch_count
                    val_loss_meter['total'] /= val_batch_count

                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, Metric_LPIPS={mean_lpips:6.4f}...')
                self.logger.info(f'Validation Loss  : Total={val_loss_meter["total"]:.4f}, MSE={val_loss_meter["mse"]:.4f}, LPIPS={val_loss_meter["lpips"]:.4f}')

                self.logging_metric(mean_psnr, tag='Metric_PSNR', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='Metric_LPIPS', phase=phase, add_global_step=False)
                
                self.logging_metric(val_loss_meter['total'], tag='Loss/Total', phase=phase, add_global_step=False)
                self.logging_metric(val_loss_meter['mse'], tag='Loss/MSE', phase=phase, add_global_step=False)
                # Only increment global step ONCE
                self.logging_metric(val_loss_meter['lpips'], tag='Loss/LPIPS', phase=phase, add_global_step=True)

            self.logger.info("="*100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

def replace_nan_in_batch(im_lq, im_gt):
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    from utils import util_image
    from  einops import rearrange
    im1 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00012685_crop000.png',
                            chn = 'rgb', dtype='float32')
    im2 = util_image.imread('./testdata/inpainting/val/places/Places365_val_00014886_crop000.png',
                            chn = 'rgb', dtype='float32')
    im = rearrange(np.stack((im1, im2), 3), 'h w c b -> b c h w')
    im_grid = im.copy()
    for alpha in [0.8, 0.4, 0.1, 0]:
        im_new = im * alpha + np.random.randn(*im.shape) * (1 - alpha)
        im_grid = np.concatenate((im_new, im_grid), 1)

    im_grid = np.clip(im_grid, 0.0, 1.0)
    im_grid = rearrange(im_grid, 'b (k c) h w -> (b k) c h w', k=5)
    xx = vutils.make_grid(torch.from_numpy(im_grid), nrow=5, normalize=True, scale_each=True).numpy()
    util_image.imshow(np.concatenate((im1, im2), 0))
    util_image.imshow(xx.transpose((1,2,0)))
