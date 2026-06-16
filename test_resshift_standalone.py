import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import logging
import sys
import importlib
import math
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from omegaconf import OmegaConf, DictConfig

# ================= 配置区域 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_TO_SAVE = 100    # [Modified] 保存前 100 张
MAX_TEST_SAMPLES = None # 设置为 None 跑完所有数据
# ===========================================
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


try:
    import lpips
except ImportError:
    print("!!! Error: lpips not found. Please run: pip install lpips")
    exit()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "test_log.txt")
    logger = logging.getLogger("ResShift_Test")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    return logger

# === 动态实例化函数 ===
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if "target" in config.get("params", {}):
            return get_obj_from_str(config["params"]["target"])(**config.get("params", dict()))
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# === [Metrics] 内置 SSIM/PSNR 计算 ===
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average: return ssim_map.mean()
    else: return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim_pt(img, img2, crop_border=0):
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    (_, channel, _, _) = img.size()
    window_size = 11
    window = create_window(window_size, channel)
    if img.is_cuda: window = window.cuda(img.get_device())
    window = window.type_as(img)
    return _ssim(img, img2, window, window_size, channel, size_average=True)

def calculate_psnr_pt(img, img2, crop_border=0):
    mse = torch.mean((img - img2) ** 2)
    if mse == 0: return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# === 模型加载 ===
def load_models(cfg, resshift_ckpt_path):
    print("[INFO] Loading Stage 1 SwinIR...")
    if isinstance(cfg.swinir, DictConfig):
        OmegaConf.set_struct(cfg.swinir, False)
    if 'target' not in cfg.swinir:
        cfg.swinir.target = 'stage1_utils.swinir.SwinIR'
        
    swinir = instantiate_from_config(cfg.swinir)
    
    if os.path.exists(cfg.swinir.ckpt_path):
        sd = torch.load(cfg.swinir.ckpt_path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        # 去 module. 前缀
        new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
        swinir.load_state_dict(new_sd, strict=True)
    else:
        print(f"[WARN] SwinIR checkpoint not found at {cfg.swinir.ckpt_path}! (Random weights used)")
    swinir.to(DEVICE).eval()

    print("[INFO] Loading VQGAN Autoencoder...")
    autoencoder = instantiate_from_config(cfg.autoencoder)
    if os.path.exists(cfg.autoencoder.ckpt_path):
        sd = torch.load(cfg.autoencoder.ckpt_path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        autoencoder.load_state_dict(sd, strict=True)
    else:
        print(f"[WARN] VQGAN checkpoint not found at {cfg.autoencoder.ckpt_path}!")
    autoencoder.to(DEVICE).eval()

    print(f"[INFO] Loading Stage 2 ResShift (UNet) from {resshift_ckpt_path}...")
    model = instantiate_from_config(cfg.model)
    
    if os.path.exists(resshift_ckpt_path):
        sd = torch.load(resshift_ckpt_path, map_location="cpu")
        # 智能识别权重类型
        if "params_ema" in sd: 
            print("[INFO] Loading EMA weights...")
            sd = sd["params_ema"]
        elif "state_dict" in sd:
            print("[INFO] Loading standard weights...")
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
            
        # 智能去除 module. 前缀
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v
        
        try:
            model.load_state_dict(new_sd, strict=True)
            print("[INFO] ResShift weights loaded successfully (Strict).")
        except RuntimeError as e:
            print(f"[WARN] Strict loading failed: {e}")
            print("[WARN] Trying lax loading...")
            model.load_state_dict(new_sd, strict=False)
    else:
        print(f"[ERROR] ResShift checkpoint not found at {resshift_ckpt_path}")
        exit()
        
    model.to(DEVICE).eval()

    print("[INFO] Creating Diffusion Sampler...")
    diffusion = instantiate_from_config(cfg.diffusion)
    
    return swinir, autoencoder, model, diffusion

# === 推理逻辑 ===
def run_inference(swinir, autoencoder, model, diffusion, lq_tensor):
    """
    输入: lq_tensor (B, 3, 128, 128) [0, 1]
    输出: output_hr (B, 3, 512, 512) [0, 1]
    """
    # 1. SwinIR 去噪 (128 -> 128)
    with torch.no_grad():
        lq_clean = swinir(lq_tensor)
        
    # 2. ResShift 超分 (128 -> 512)
    cond_norm = (lq_clean - 0.5) * 2.0
    model_kwargs = {'lq': cond_norm}
    
    latent_size = 512 // 8
    noise = torch.randn((lq_tensor.shape[0], 8, latent_size, latent_size), device=DEVICE)
    
    with torch.no_grad():
        # [Fix] 移除 shape 参数，添加 y 参数
        out_latent = diffusion.p_sample_loop(
            y=cond_norm,       # <--- Added explicit y
            model=model, 
            # shape=noise.shape, # <--- Removed
            noise=noise, 
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            first_stage_model=autoencoder, 
            progress=False, 
            device=DEVICE
        )
        
        if out_latent.shape[1] == 8:
            out_hr = autoencoder.decode(out_latent)
        else:
            out_hr = out_latent

        out_hr = (out_hr + 1.0) / 2.0
        out_hr = torch.clamp(out_hr, 0, 1)
        
    return out_hr, lq_clean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Training config path")
    parser.add_argument("--ckpt", type=str, required=True, help="ResShift checkpoint path")
    parser.add_argument("--save_dir", type=str, default="results_resshift", help="Output directory")
    args = parser.parse_args()

    set_seed(123456)
    logger = setup_logger(args.save_dir)
    
    # 1. Load Config
    logger.info(f"Loading Config: {args.config}")
    cfg = OmegaConf.load(args.config)
    
    # 2. Load Dataset
    val_dataset_cfg = None
    if 'data' in cfg and 'val' in cfg.data: val_dataset_cfg = cfg.data.val
    elif 'dataset' in cfg and 'val' in cfg.dataset: val_dataset_cfg = cfg.dataset.val
    
    if val_dataset_cfg is None:
        logger.error("Error: Cannot find validation dataset config!")
        return
        
    try:
        from torch.utils.data import DataLoader
        val_dataset = instantiate_from_config(val_dataset_cfg)
        # [Critical] num_workers=0 强制在主进程加载数据
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Dataset loaded (num_workers=0). Total samples: {len(val_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 3. Load Models
    swinir, autoencoder, model, diffusion = load_models(cfg, args.ckpt)
    
    # 4. Init LPIPS
    logger.info("Initializing LPIPS (VGG)...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(DEVICE).eval()

    # 5. Testing Loop
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    count = 0
    saved_count = 0
    
    logger.info("Starting Inference...")
    
    # [Critical] 重置种子
    set_seed(123456)

    for i, batch in enumerate(tqdm(val_loader)):
        if MAX_TEST_SAMPLES and count >= MAX_TEST_SAMPLES:
            break
            
        if isinstance(batch, dict):
            gt = batch['gt'].to(DEVICE)
            lq = batch['lq'].to(DEVICE)
        else:
            gt, lq = batch[0].to(DEVICE), batch[1].to(DEVICE)

        if gt.ndim == 5: gt = gt.squeeze(0); lq = lq.squeeze(0)

        gt_metric = (gt + 1.0) / 2.0 if gt.min() < 0 else gt
        gt_metric = torch.clamp(gt_metric, 0, 1)
        
        lq_input = (lq + 1.0) / 2.0 if lq.min() < 0 else lq
        lq_input = torch.clamp(lq_input, 0, 1)

        # === Run Inference ===
        try:
            output_hr, lq_clean = run_inference(swinir, autoencoder, model, diffusion, lq_input)
        except Exception as e:
            logger.error(f"Inference failed at index {i}: {e}")
            continue

        if output_hr.shape[-1] != gt_metric.shape[-1]:
             output_hr = F.interpolate(output_hr, size=gt_metric.shape[-2:], mode='bicubic')

        p = calculate_psnr_pt(output_hr, gt_metric).mean().item()
        l = loss_fn_lpips(output_hr, gt_metric, normalize=True).mean().item()
        
        s = calculate_ssim_pt(output_hr, gt_metric, crop_border=0)
        if torch.is_tensor(s): s = s.item()

        total_psnr += p
        total_lpips += l
        total_ssim += s
        count += 1
        
        if saved_count < IMAGES_TO_SAVE:
            lq_viz = F.interpolate(lq_input, size=gt_metric.shape[-2:], mode='nearest')
            swinir_viz = F.interpolate(lq_clean, size=gt_metric.shape[-2:], mode='nearest')
            
            grid = make_grid([
                gt_metric[0], 
                lq_viz[0], 
                swinir_viz[0], 
                output_hr[0]
            ], nrow=4, padding=0)
            
            # [Added] Filename with SSIM
            save_path = os.path.join(args.save_dir, f"res_{i:04d}_P{p:.2f}_S{s:.3f}_L{l:.3f}.png")
            save_image(grid, save_path)
            saved_count += 1

    if count > 0:
        logger.info("="*40)
        logger.info(f"ResShift Test Results ({count} samples)")
        logger.info("="*40)
        logger.info(f"PSNR : {total_psnr / count:.4f}")
        logger.info(f"SSIM : {total_ssim / count:.4f}")
        logger.info(f"LPIPS: {total_lpips / count:.4f}")
        logger.info("="*40)
    else:
        logger.info("No images processed.")

if __name__ == "__main__":
    main()