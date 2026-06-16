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

# ================= 配置区域 =================
IMAGES_TO_SAVE = 10   # 建议只保存少量图片用于肉眼对比
MAX_TEST_SAMPLES = 100 
BATCH_SIZE = 1        # 保持为 1 以便控制流程
DEVICE = "cuda"
# ===========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, "ablation_log_resshift.txt")
    logger = logging.getLogger("AblationTestResShift")
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

# === Metrics ===
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

# === 退化逻辑 (模拟低清图) ===
def apply_degradation_to_tensor(gt_tensor, params):
    """GT (512) [-1,1] -> LQ (128) [0,1]"""
    # GT is [-1, 1] tensor -> [0, 255] numpy
    imgs_np = ((gt_tensor.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2 * 255.0).clip(0, 255).astype(np.uint8)
    lq_list = []
    
    p_blur = params.get('blur_sigma', 0)
    p_scale = params.get('downsample_range', 1)
    p_noise = params.get('noise_range', 0)
    p_jpeg = params.get('jpeg_range', 100)
    
    for img in imgs_np:
        # 1. Base Resize 512 -> 128 
        img_lq = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        # 2. Blur
        if p_blur > 0.1:
            k_size = int(p_blur * 4) + 1
            if k_size % 2 == 0: k_size += 1
            k_size = max(3, k_size)
            img_lq = cv2.GaussianBlur(img_lq, (k_size, k_size), p_blur)
            
        # 3. Downsample & Upsample 
        if p_scale > 1.0:
            h, w = img_lq.shape[:2]
            new_h, new_w = int(h / p_scale), int(w / p_scale)
            img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
            
        # 4. Noise
        if p_noise > 0:
            noise = np.random.normal(0, p_noise, img_lq.shape)
            img_lq = img_lq.astype(np.float32) + noise
            img_lq = np.clip(img_lq, 0, 255).astype(np.uint8)
            
        # 5. JPEG
        if p_jpeg < 100:
            img_bgr = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
            _, encimg = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(p_jpeg)])
            img_bgr = cv2.imdecode(encimg, 1)
            img_lq = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        lq_list.append(img_lq)
        
    lq_np = np.array(lq_list)
    # [0, 255] -> [0, 1] tensor
    lq_tensor = torch.from_numpy(lq_np).float() / 255.0
    lq_tensor = lq_tensor.permute(0, 3, 1, 2)
    return lq_tensor

def get_base_config_params():
    return {
        'blur_sigma': 0.1,       
        'downsample_range': 1.0,     
        'noise_range': 0,          
        'jpeg_range': 100 
    }

# === 模型加载 ===
def load_full_pipeline(cfg, stage2_ckpt_path):
    print("[INFO] Loading Stage 1 SwinIR...")
    if 'target' not in cfg.swinir:
        cfg.swinir.target = 'stage1_utils.swinir.SwinIR'
    swinir = instantiate_from_config(cfg.swinir)
    # Load SwinIR Weights
    if os.path.exists(cfg.swinir.ckpt_path):
        sd = torch.load(cfg.swinir.ckpt_path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        swinir.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)
    else:
        print(f"[WARN] SwinIR checkpoint not found at {cfg.swinir.ckpt_path}, using random weights.")
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
    # Load Stage 2 Weights
    sd = torch.load(stage2_ckpt_path, map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v
        
    try:
        model.load_state_dict(new_sd, strict=True)
        print("[INFO] ResShift weights loaded successfully (Strict).")
    except Exception as e:
        print(f"[WARN] Strict loading failed: {e}")
        print("[WARN] Trying lax loading (missing keys might be expected if structure differs slightly)...")
        model.load_state_dict(new_sd, strict=False)
        
    model.to(DEVICE).eval()

    print("[INFO] Creating Diffusion Sampler...")
    diffusion = instantiate_from_config(cfg.diffusion)
    
    return swinir, autoencoder, model, diffusion

def run_resshift_inference(
    swinir, autoencoder, model, diffusion, 
    lq_tensor, # [0, 1]
    cfg
):
    """
    执行完整的推理流程：
    1. LQ (128) -> SwinIR -> Clean LQ (128)
    2. Clean LQ -> ResShift (UNet + Diffusion) -> Latent Z0
    3. Latent Z0 -> VQGAN Decoder -> HR (512)
    """
    # 1. SwinIR Denoising
    # SwinIR 期望输入 [0, 1]
    with torch.no_grad():
        # 这里 SwinIR 输出是 [0, 1]
        lq_clean = swinir(lq_tensor) 
    
    # 2. Prepare Conditions for ResShift
    # ResShift 训练时 lq 输入范围通常是 [-1, 1] 
    lq_clean_norm = (lq_clean - 0.5) * 2.0
    
    model_kwargs = {'lq': lq_clean_norm}
    
    # 3. Prepare Noise
    # VQGAN f8 -> 512 / 8 = 64
    latent_size = 512 // 8
    noise_shape = (lq_tensor.shape[0], 8, latent_size, latent_size) # 8 channels from VQGAN config
    noise = torch.randn(noise_shape, device=DEVICE)
    
    # 4. Diffusion Sampling
    with torch.no_grad():

        sample_fn = diffusion.p_sample_loop
        
        output_hr = sample_fn(
            y=lq_clean_norm,                # <--- Explicit y (cond)
            model=model,                    # <--- Explicit model
            noise=noise,
            clip_denoised=False, 
            model_kwargs=model_kwargs,
            first_stage_model=autoencoder,  # <--- Added first_stage_model
            progress=False,
            device=DEVICE
        )
        
        # 5. Decode Latent to Image
        if output_hr.shape[1] == 8: # It is latent
             output_hr = autoencoder.decode(output_hr)
        
        # [-1, 1] -> [0, 1]
        output_hr = (output_hr + 1.0) / 2.0
        output_hr = torch.clamp(output_hr, 0, 1)
        
    return lq_clean, output_hr

def run_evaluation_loop(
    swinir, autoencoder, model, diffusion, val_loader, loss_fn_lpips, 
    save_dir, param_name, param_val, cfg
):
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_count = 0
    global_img_idx = 0
    
    current_params = get_base_config_params()
    current_params[param_name] = param_val
    
    pbar = tqdm(val_loader, desc=f"Eval {param_name}={param_val}", leave=False)
    
    for i, batch in enumerate(pbar):
        if total_count >= MAX_TEST_SAMPLES:
            break

        if isinstance(batch, dict):
            gt = batch['gt'].to(DEVICE)
        else:
            gt, _ = batch[0].to(DEVICE), batch[1].to(DEVICE)
            
        if gt.ndim == 5: gt = gt.squeeze(0)
        
        # Apply degradation (GT 512 -> LQ 128)
        # GT [-1, 1], LQ [0, 1]
        lq = apply_degradation_to_tensor(gt, current_params).to(DEVICE)
        
        # GT Normalize for metric: [-1, 1] -> [0, 1]
        gt_metric = (gt + 1.0) / 2.0
        gt_metric = torch.clamp(gt_metric, 0, 1)
        
        # Run Inference
        swinir_out, resshift_out = run_resshift_inference(
            swinir, autoencoder, model, diffusion, lq, cfg
        )
        
        # Calculate Metrics (Compare ResShift Output 512 vs GT 512)
        psnr_val = calculate_psnr_pt(resshift_out, gt_metric, crop_border=0).mean().item()
        ssim_val = calculate_ssim_pt(resshift_out, gt_metric, crop_border=0)
        if torch.is_tensor(ssim_val): ssim_val = ssim_val.item()
        lpips_val = loss_fn_lpips(resshift_out, gt_metric, normalize=True).mean().item()

        current_bs = gt.shape[0]
        total_psnr += psnr_val * current_bs
        total_ssim += ssim_val * current_bs
        total_lpips += lpips_val * current_bs
        total_count += current_bs

        if save_dir:
            for b_idx in range(current_bs):
                if global_img_idx < IMAGES_TO_SAVE:
                    img_filename = f"img_{global_img_idx:04d}_P{psnr_val:.2f}_S{ssim_val:.3f}_L{lpips_val:.3f}.png"
                    save_path = os.path.join(save_dir, img_filename)
                    
                    # Resize LQ/SwinIR to 512 for visualization
                    lq_viz = F.interpolate(lq[b_idx:b_idx+1], size=(512, 512), mode='nearest')
                    swinir_viz = F.interpolate(swinir_out[b_idx:b_idx+1], size=(512, 512), mode='nearest')
                    
                    # Grid: GT | Input(LQ) | SwinIR | ResShift
                    grid = make_grid([
                        gt_metric[b_idx], 
                        lq_viz[0], 
                        swinir_viz[0], 
                        resshift_out[b_idx]
                    ], nrow=4, padding=0)
                    save_image(grid, save_path)
                    global_img_idx += 1
                else:
                    break
    
    # Avoid division by zero if something went wrong
    if total_count == 0: return 0, 0, 0

    return total_psnr / total_count, total_ssim / total_count, total_lpips / total_count

def plot_charts(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics_config = {
        'PSNR': {'color': '#1f77b4', 'marker': 'o', 'ylabel': 'PSNR (dB)'},
        'SSIM': {'color': '#2ca02c', 'marker': 's', 'ylabel': 'SSIM'},
        'LPIPS': {'color': '#d62728', 'marker': '^', 'ylabel': 'LPIPS (Lower is Better)'}
    }

    for param_name, data in results.items():
        x_vals = data['x']
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1); plt.plot(x_vals, data['PSNR'], marker='o', color='#1f77b4'); plt.title(f"{param_name} vs PSNR"); plt.grid(True, alpha=0.5)
        plt.subplot(1, 3, 2); plt.plot(x_vals, data['SSIM'], marker='s', color='#2ca02c'); plt.title(f"{param_name} vs SSIM"); plt.grid(True, alpha=0.5)
        plt.subplot(1, 3, 3); plt.plot(x_vals, data['LPIPS'], marker='^', color='#d62728'); plt.title(f"{param_name} vs LPIPS"); plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"chart_{param_name}_combined.png"), dpi=300)
        plt.close()

    levels = range(1, 11)
    markers = {'blur_sigma': 'o', 'noise_range': 's', 'downsample_range': '^', 'jpeg_range': 'D'}
    plt.figure(figsize=(18, 6))
    metric_keys = ['PSNR', 'SSIM', 'LPIPS']
    for idx, metric in enumerate(metric_keys):
        plt.subplot(1, 3, idx+1)
        for name, data in results.items():
            plt.plot(levels, data[metric], marker=markers.get(name, 'o'), linewidth=2, label=name)
        plt.title(f"Impact on {metric} (Normalized)", fontsize=12)
        plt.xlabel("Severity Level"); plt.ylabel(metric); plt.legend(); plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart_summary_combined.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Stage 2 配置文件 (.yaml)")
    parser.add_argument("--ckpt_stage2", type=str, required=True, help="ResShift 权重文件 (.pth)")
    parser.add_argument("--save_dir", type=str, default="ablation_results_resshift", help="结果保存目录")
    args = parser.parse_args()

    set_seed(123456)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)
    
    logger.info(f"Loading Config: {args.config}")
    cfg = OmegaConf.load(args.config)
    
    # 1. 加载所有模型 (SwinIR, VQGAN, ResShift)
    swinir, autoencoder, model, diffusion = load_full_pipeline(cfg, args.ckpt_stage2)
    
    # 2. 初始化 LPIPS
    logger.info("Initializing LPIPS (VGG)...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(DEVICE).eval()

    # 3. 实验参数 (与 Stage 1 对齐)
    experiments = {
        'blur_sigma': np.round(np.linspace(0.1, 4.0, 10), 2).tolist(), 
        'noise_range': np.round(np.linspace(0, 30, 10), 2).tolist(),
        'downsample_range': np.round(np.linspace(1.0, 4.0, 10), 2).tolist(),
        'jpeg_range': np.round(np.linspace(100, 10, 10)).astype(int).tolist()
    }

    all_results = {}
    
    # 4. 准备 Dataset (只用来读 GT)
    # 从 config 中读取 data.val 的配置
    temp_cfg = cfg.copy()
    val_dataset_cfg = None
    if 'data' in temp_cfg and 'val' in temp_cfg.data:
        val_dataset_cfg = temp_cfg.data.val
    elif 'dataset' in temp_cfg and 'val' in temp_cfg.dataset:
        val_dataset_cfg = temp_cfg.dataset.val
        
    if val_dataset_cfg is None:
        logger.error("Config Error: Cannot find validation dataset params.")
        return

    # 实例化 Dataset 
    val_dataset = instantiate_from_config(val_dataset_cfg)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 5. 开始循环测试
    for param_key, test_values in experiments.items():
        logger.info(f"\n{'='*20} Testing Parameter: {param_key} {'='*20}")
        param_root_dir = os.path.join(args.save_dir, param_key)
        os.makedirs(param_root_dir, exist_ok=True)
        metric_history = {'x': [], 'PSNR': [], 'SSIM': [], 'LPIPS': []}
        
        for val in test_values:
            sub_dir_name = f"val_{val}"
            current_img_save_dir = os.path.join(param_root_dir, sub_dir_name)
            os.makedirs(current_img_save_dir, exist_ok=True)
            
            # 运行评估
            avg_psnr, avg_ssim, avg_lpips = run_evaluation_loop(
                swinir, autoencoder, model, diffusion, val_loader, loss_fn_lpips, 
                current_img_save_dir, param_key, val, cfg
            )
            
            logger.info(f"    [{param_key}={val}] -> PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
            metric_history['x'].append(val)
            metric_history['PSNR'].append(avg_psnr)
            metric_history['SSIM'].append(avg_ssim)
            metric_history['LPIPS'].append(avg_lpips)
        
        all_results[param_key] = metric_history

    logger.info("\nGenerating Charts...")
    plot_charts(all_results, args.save_dir)
    logger.info(f"Done! Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
