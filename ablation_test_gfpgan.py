import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import lpips
import cv2
import logging 
import importlib
import math
import sys


try:
    import basicsr
    if not hasattr(basicsr, 'archs'):
        print("[INFO] BasicSR version > 1.4.2 detected. Applying compatibility patch...")
        try:
            import basicsr.models
        except ImportError:
            pass
        if hasattr(basicsr, 'models'):
            sys.modules['basicsr.archs'] = basicsr.models
            basicsr.archs = basicsr.models
            print("[INFO] Patch applied: basicsr.archs -> basicsr.models")
        else:
            print("[WARN] Could not find basicsr.models. GFPGAN might fail.")
except ImportError as e:
    print(f"[WARN] BasicSR import failed: {e}")
    print("If you haven't installed it: pip install basicsr")

# ==============================================================================

# === GFPGAN Import ===
try:
    from gfpgan import GFPGANer
except ImportError as e:
    import traceback
    traceback.print_exc()
    print("="*60)
    print(f"!!! Import Error: {e}")
    print("!!! Potential Fixes:")
    print("1. pip install basicsr facexlib")
    print("2. pip install --upgrade gfpgan")
    print("3. Ensure you are running in the correct conda environment.")
    print("="*60)
    exit()

# ================= 配置区域 =================
IMAGES_TO_SAVE = 20    # 保存图片数量
MAX_TEST_SAMPLES = 200 # 测试多少张图片后停止
BATCH_SIZE = 1         # 建议保持为 1
DEVICE = "cuda"
# ===========================================

# === 基础工具函数 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, "comparison_log.txt")
    logger = logging.getLogger("ComparisonTest")
    logger.setLevel(logging.INFO)
    logger.handlers = [] 
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger

# === 动态导入辅助函数 ===
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

# === Metrics (PSNR/SSIM) ===
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

def load_our_pipeline(cfg, stage2_ckpt_path):
    print("[INFO] Loading Stage 1 SwinIR...")
    if 'target' not in cfg.swinir:
        cfg.swinir.target = 'stage1_utils.swinir.SwinIR'
    swinir = instantiate_from_config(cfg.swinir)
    if os.path.exists(cfg.swinir.ckpt_path):
        sd = torch.load(cfg.swinir.ckpt_path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        swinir.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)
    else:
        print(f"[WARN] SwinIR checkpoint not found at {cfg.swinir.ckpt_path}!")
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

    print(f"[INFO] Loading Stage 2 ResShift (UNet) from {stage2_ckpt_path}...")
    model = instantiate_from_config(cfg.model)
    sd = torch.load(stage2_ckpt_path, map_location="cpu")
    
    if "ema" in stage2_ckpt_path or "params_ema" in sd:
        print("[INFO] Trying to load EMA weights...")
        if "params_ema" in sd: sd = sd["params_ema"]
    elif "state_dict" in sd: 
        sd = sd["state_dict"]
    
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."): k = k[7:]
        new_sd[k] = v
        
    try:
        model.load_state_dict(new_sd, strict=True)
        print("[INFO] ResShift weights loaded successfully (Strict).")
    except Exception as e:
        print(f"[WARN] Strict loading failed: {e}. Trying lax loading...")
        model.load_state_dict(new_sd, strict=False)
        
    model.to(DEVICE).eval()
    print("[INFO] Creating Diffusion Sampler...")
    diffusion = instantiate_from_config(cfg.diffusion)
    
    return swinir, autoencoder, model, diffusion

# === 加载 GFPGAN ===
def load_gfpgan(model_path):
    print(f"[INFO] Loading GFPGAN from {model_path}...")
    # upscale=4: 输入 128 -> 输出 512
    restorer = GFPGANer(
        model_path=model_path,
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device=torch.device(DEVICE)
    )
    return restorer

def run_ours_inference(swinir, autoencoder, model, diffusion, lq_tensor):
    # 1. SwinIR Denoising (128 -> 128)
    with torch.no_grad():
        lq_clean = swinir(lq_tensor)
    
    # 2. ResShift Super-Res (128 -> 512)
    cond_norm = (lq_clean - 0.5) * 2.0
    model_kwargs = {'lq': cond_norm}
    
    # Latent size for 512px image with f8 VQGAN is 64
    latent_size = 512 // 8 
    noise = torch.randn((lq_tensor.shape[0], 8, latent_size, latent_size), device=DEVICE)
    
    with torch.no_grad():
        out_latent = diffusion.p_sample_loop(
            model, noise.shape, noise, clip_denoised=False, 
            model_kwargs=model_kwargs, first_stage_model=autoencoder, 
            progress=False, device=DEVICE
        )
        if out_latent.shape[1] == 8:
            out_hr = autoencoder.decode(out_latent)
        else:
            out_hr = out_latent

        out_hr = (out_hr + 1.0) / 2.0
        out_hr = torch.clamp(out_hr, 0, 1)
    return out_hr

def run_gfpgan_inference(gfpganer, lq_tensor):
    batch_res = []
    lq_np = (lq_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    for img in lq_np:
        # RGB -> BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        _, _, output_bgr = gfpganer.enhance(
            img_bgr, 
            has_aligned=True, # 假设是人脸裁切图
            only_center_face=False, 
            paste_back=False
        )
        
        # BGR -> RGB
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        batch_res.append(output_rgb)
        
    # Stack back to Tensor [0, 1]
    res_np = np.array(batch_res)
    res_tensor = torch.from_numpy(res_np).float() / 255.0
    res_tensor = res_tensor.permute(0, 3, 1, 2).to(DEVICE)
    
    return res_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="My Stage 2 Config (.yaml)")
    parser.add_argument("--ckpt_stage2", type=str, required=True, help="My ResShift Weights (.pth)")
    parser.add_argument("--gfpgan_ckpt", type=str, default="weights/GFPGANv1.4.pth", help="GFPGAN Weights")
    parser.add_argument("--save_dir", type=str, default="results_comparison_gfpgan", help="Output Dir")
    args = parser.parse_args()

    set_seed(123456)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)
    
    # 1. Load Our Models
    logger.info(f"Loading Config: {args.config}")
    cfg = OmegaConf.load(args.config)
    swinir, autoencoder, model, diffusion = load_our_pipeline(cfg, args.ckpt_stage2)
    
    # 2. Load GFPGAN
    gfpgan_model = load_gfpgan(args.gfpgan_ckpt)

    # 3. Metrics
    logger.info("Initializing LPIPS (VGG)...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(DEVICE).eval()

    # 4. Load Dataset from Config
    val_dataset_cfg = None
    if 'data' in cfg and 'val' in cfg.data: val_dataset_cfg = cfg.data.val
    elif 'dataset' in cfg and 'val' in cfg.dataset: val_dataset_cfg = cfg.dataset.val
    
    if val_dataset_cfg is None:
        logger.error("Cannot find validation dataset config!")
        return

    logger.info(f"Loading Validation Dataset...")
    val_dataset = instantiate_from_config(val_dataset_cfg)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 5. Testing Loop
    metrics = {
        'Ours': {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0},
        'GFPGAN': {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
    }
    
    count = 0
    img_idx = 0
    logger.info(f"Starting Comparison Loop (Max {MAX_TEST_SAMPLES} samples)...")
    
    for i, batch in enumerate(tqdm(val_loader)):
        if MAX_TEST_SAMPLES and count >= MAX_TEST_SAMPLES: break
        
        # Get Data
        if isinstance(batch, dict):
            gt = batch['gt'].to(DEVICE)
            lq = batch['lq'].to(DEVICE) 
        else:
            # CodeFormerDataset 返回: gt, lq, prompt
            gt, lq = batch[0].to(DEVICE), batch[1].to(DEVICE)

        if gt.ndim == 5: gt = gt.squeeze(0); lq = lq.squeeze(0)

        gt_metric = (gt + 1.0) / 2.0 if gt.min() < 0 else gt
        gt_metric = torch.clamp(gt_metric, 0, 1)
        
        # Normalize LQ to [0, 1] for GFPGAN/SwinIR
        lq_input = (lq + 1.0) / 2.0 if lq.min() < 0 else lq
        lq_input = torch.clamp(lq_input, 0, 1)

        current_bs = gt.shape[0]

        # === Run Inference ===
        # A. Ours
        hr_ours = run_ours_inference(swinir, autoencoder, model, diffusion, lq_input)
        
        # B. GFPGAN
        hr_gfpgan = run_gfpgan_inference(gfpgan_model, lq_input)
        
        # === Metrics Calculation ===
        def compute_metrics(pred, target):
            # Ensure size matches (GFPGAN output might vary slightly if not aligned?)
            if pred.shape[-1] != target.shape[-1]:
                pred = F.interpolate(pred, size=target.shape[-2:], mode='bicubic')
            
            p = calculate_psnr_pt(pred, target, crop_border=0).mean().item()
            s = calculate_ssim_pt(pred, target, crop_border=0)
            if torch.is_tensor(s): s = s.item()
            l = loss_fn_lpips(pred, target, normalize=True).mean().item()
            return p, s, l

        p1, s1, l1 = compute_metrics(hr_ours, gt_metric)
        p2, s2, l2 = compute_metrics(hr_gfpgan, gt_metric)
        
        metrics['Ours']['psnr'] += p1 * current_bs
        metrics['Ours']['ssim'] += s1 * current_bs
        metrics['Ours']['lpips'] += l1 * current_bs
        
        metrics['GFPGAN']['psnr'] += p2 * current_bs
        metrics['GFPGAN']['ssim'] += s2 * current_bs
        metrics['GFPGAN']['lpips'] += l2 * current_bs
        
        count += current_bs

        # === Save Images ===
        if img_idx < IMAGES_TO_SAVE:
            # LQ resize for visualization
            lq_viz = F.interpolate(lq_input, size=(512, 512), mode='nearest')
            # Grid: GT | LQ | Ours | GFPGAN
            grid = make_grid([
                gt_metric[0], lq_viz[0], hr_ours[0], hr_gfpgan[0]
            ], nrow=4, padding=0)
            
            save_name = f"img_{img_idx:03d}_Ours(P{p1:.1f}_L{l1:.2f})_GFP(P{p2:.1f}_L{l2:.2f}).png"
            save_image(grid, os.path.join(args.save_dir, save_name))
            img_idx += 1

    # === Final Report ===
    if count > 0:
        logger.info("="*40)
        logger.info(f"Comparison Result (Avg over {count} images)")
        logger.info("="*40)
        for method in metrics:
            p = metrics[method]['psnr'] / count
            s = metrics[method]['ssim'] / count
            l = metrics[method]['lpips'] / count
            logger.info(f"{method:10s} | PSNR: {p:.2f} | SSIM: {s:.4f} | LPIPS: {l:.4f}")
        logger.info("="*40)
        logger.info(f"Images saved to: {args.save_dir}")

if __name__ == "__main__":
    main()