import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import lpips
import cv2
import logging 
import importlib
import math

# ================= 配置区域 =================
IMAGES_TO_SAVE = 20
BATCH_SIZE = 1  
# ===========================================

# === 内置 SSIM 计算逻辑 ===
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
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim_pt(img, img2, crop_border=0, test_y_channel=False):
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if test_y_channel and img.shape[1] == 3:
        img = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        img2 = 0.299 * img2[:, 0, :, :] + 0.587 * img2[:, 1, :, :] + 0.114 * img2[:, 2, :, :]
        img = img.unsqueeze(1)
        img2 = img2.unsqueeze(1)
    (_, channel, _, _) = img.size()
    window_size = 11
    window = create_window(window_size, channel)
    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)
    return _ssim(img, img2, window, window_size, channel, size_average=True)

try:
    from stage1_utils.common import instantiate_from_config, calculate_psnr_pt
    try:
        from stage1_utils.common import calculate_ssim_pt
    except ImportError:
        pass 
except ImportError:
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

    try:
        from stage1_utils.common import calculate_psnr_pt
    except ImportError:
        def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
            mse = torch.mean((img - img2) ** 2)
            if mse == 0: return torch.tensor(100.0)
            return 20 * torch.log10(1.0 / torch.sqrt(mse))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, "ablation_log.txt")
    logger = logging.getLogger("AblationTest")
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
    logger.info(f"日志文件已创建: {log_file}")
    return logger


def apply_degradation_to_tensor(gt_tensor, params):
    """GT [-1,1] -> LQ [0,1]"""
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
    lq_tensor = torch.from_numpy(lq_np).float() / 255.0
    lq_tensor = lq_tensor.permute(0, 3, 1, 2)
    return lq_tensor

def get_base_config_params():
    """基础参数: 所有退化效果最小"""
    return {
        'blur_sigma': 0.1,       
        'downsample_range': 1.0,     
        'noise_range': 0,          
        'jpeg_range': 100 
    }

def run_evaluation(model, val_loader, loss_fn_lpips, device, current_img_save_dir, param_name, param_val):
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_count = 0
    global_img_idx = 0
    
    current_params = get_base_config_params()
    current_params[param_name] = param_val
    
    pbar = tqdm(val_loader, desc=f"Eval {param_name}={param_val}", leave=False)
    
    for i, batch in enumerate(pbar):
        if isinstance(batch, dict):
            gt = batch['gt'].to(device)
        else:
            gt, _ = batch[0].to(device), batch[1].to(device) # Ignore original LQ

        if gt.ndim == 5: gt = gt.squeeze(0)
        
        # Apply degradation locally
        lq = apply_degradation_to_tensor(gt, current_params).to(device)
        
        gt_metric = (gt + 1.0) / 2.0
        gt_metric = torch.clamp(gt_metric, 0, 1)
        if gt_metric.shape[1] != 3: 
            gt_metric = rearrange(gt_metric, "b h w c -> b c h w")
        gt_metric = gt_metric.contiguous().float()

        current_batch_size = gt.size(0)

        with torch.no_grad():
            pred = model(lq)
            pred = torch.clamp(pred, 0, 1)

        if pred.shape[-2:] != gt_metric.shape[-2:]:
            gt_metric_resized = F.interpolate(gt_metric, size=pred.shape[-2:], mode='bicubic', align_corners=False)
        else:
            gt_metric_resized = gt_metric

        psnr_val = calculate_psnr_pt(pred, gt_metric_resized, crop_border=0).mean().item()
        ssim_val = calculate_ssim_pt(pred, gt_metric_resized, crop_border=0)
        if torch.is_tensor(ssim_val): ssim_val = ssim_val.item()
        lpips_val = loss_fn_lpips(pred, gt_metric_resized, normalize=True).mean().item()

        total_psnr += psnr_val * current_batch_size
        total_ssim += ssim_val * current_batch_size
        total_lpips += lpips_val * current_batch_size
        total_count += current_batch_size

        if current_img_save_dir:
            for b_idx in range(current_batch_size):
                if global_img_idx < IMAGES_TO_SAVE:
                    img_filename = f"img_{global_img_idx:04d}_P{psnr_val:.2f}_S{ssim_val:.3f}_L{lpips_val:.3f}.png"
                    save_path = os.path.join(current_img_save_dir, img_filename)
                    lq_viz = F.interpolate(lq[b_idx:b_idx+1], size=gt_metric_resized.shape[-2:], mode='nearest')
                    grid = make_grid([gt_metric_resized[b_idx], lq_viz[0], pred[b_idx]], nrow=3, padding=0)
                    save_image(grid, save_path)
                    global_img_idx += 1
                else:
                    break

    avg_psnr = total_psnr / total_count
    avg_ssim = total_ssim / total_count
    avg_lpips = total_lpips / total_count
    return avg_psnr, avg_ssim, avg_lpips

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
        plt.subplot(1, 3, 1)
        plt.plot(x_vals, data['PSNR'], marker='o', color='#1f77b4', linewidth=2)
        plt.title(f"{param_name} vs PSNR", fontsize=12, fontweight='bold')
        plt.xlabel("Parameter Value")
        plt.ylabel("PSNR (dB)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 3, 2)
        plt.plot(x_vals, data['SSIM'], marker='s', color='#2ca02c', linewidth=2)
        plt.title(f"{param_name} vs SSIM", fontsize=12, fontweight='bold')
        plt.xlabel("Parameter Value")
        plt.ylabel("SSIM")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 3, 3)
        plt.plot(x_vals, data['LPIPS'], marker='^', color='#d62728', linewidth=2)
        plt.title(f"{param_name} vs LPIPS", fontsize=12, fontweight='bold')
        plt.xlabel("Parameter Value")
        plt.ylabel("LPIPS (Lower is Better)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"chart_{param_name}_combined.png"), dpi=300)
        plt.close()

        for metric, cfg in metrics_config.items():
            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, data[metric], marker=cfg['marker'], color=cfg['color'], linewidth=2)
            plt.title(f"{param_name} - {metric}", fontsize=14, fontweight='bold')
            plt.xlabel(param_name, fontsize=12)
            plt.ylabel(cfg['ylabel'], fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"chart_{param_name}_{metric}.png"), dpi=300)
            plt.close()

    levels = range(1, 11)
    markers = {'blur_sigma': 'o', 'noise_range': 's', 'downsample_range': '^', 'jpeg_range': 'D'}

    plt.figure(figsize=(18, 6))
    metric_keys = ['PSNR', 'SSIM', 'LPIPS']
    for idx, metric in enumerate(metric_keys):
        plt.subplot(1, 3, idx+1)
        for name, data in results.items():
            plt.plot(levels, data[metric], marker=markers.get(name, 'o'), linewidth=2, label=name)
        plt.title(f"Impact on {metric} (Normalized)", fontsize=12, fontweight='bold')
        plt.xlabel("Degradation Severity (1=Mild, 10=Severe)")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart_summary_combined.png"), dpi=300)
    plt.close()

    for metric, cfg in metrics_config.items():
        plt.figure(figsize=(10, 6))
        for name, data in results.items():
            plt.plot(levels, data[metric], marker=markers.get(name, 'o'), linewidth=2, label=name)
        plt.title(f"Overall Impact on {metric}", fontsize=14, fontweight='bold')
        plt.xlabel("Degradation Severity (1=Mild, 10=Severe)", fontsize=12)
        plt.ylabel(cfg['ylabel'], fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"chart_summary_{metric}.png"), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="训练配置文件路径 (.yaml)")
    parser.add_argument("--ckpt", type=str, required=True, help="SwinIR 权重文件路径 (.pt)")
    parser.add_argument("--save_dir", type=str, default="ablation_results", help="结果保存根目录")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.save_dir)
    
    logger.info(f"正在加载模型: {args.ckpt}")
    cfg = OmegaConf.load(args.config)
    
    if 'swinir' in cfg:
        model_cfg = cfg.swinir
    elif 'model' in cfg and 'swinir' in cfg.model:
        model_cfg = cfg.model.swinir
    else:
        logger.warning("Can't find 'swinir' config block. Trying to use root config.")
        model_cfg = cfg

    if isinstance(model_cfg, DictConfig):
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

    if 'target' not in model_cfg:
        logger.info("[Auto-Fix] 配置缺少 'target'，正在自动注入: stage1_utils.swinir.SwinIR")
        model_cfg['target'] = 'stage1_utils.swinir.SwinIR'

    model = instantiate_from_config(model_cfg)
    
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in checkpoint: checkpoint = checkpoint["state_dict"]
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device).eval()

    # [Update] Change LPIPS backbone to VGG
    logger.info("初始化 LPIPS (VGG)...")
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()

    experiments = {
        # Blur: 0.1 ~ 4.0, 10 points
        'blur_sigma': np.round(np.linspace(0.1, 4.0, 10), 2).tolist(), 
        
        # Noise: 0 ~ 50, 10 points
        'noise_range': np.round(np.linspace(0, 30, 10), 2).tolist(),
        
        # Resize: 1.0 ~ 8.0, 10 points
        'downsample_range': np.round(np.linspace(1.0, 4.0, 10), 2).tolist(),
        
        # JPEG: 100 ~ 10, 10 points
        'jpeg_range': np.round(np.linspace(100, 10, 10)).astype(int).tolist()
    }

    all_results = {}
    logger.info(f"开始消融实验，图片保存限制: 每组 {IMAGES_TO_SAVE} 张")

    # 准备 dataset
    temp_cfg = cfg.copy()
    val_dataset_cfg = None
    if 'dataset' in temp_cfg and 'val' in temp_cfg.dataset:
        val_dataset_cfg = temp_cfg.dataset.val
    elif 'data' in temp_cfg and 'val' in temp_cfg.data:  
        val_dataset_cfg = temp_cfg.data.val
    elif 'val' in temp_cfg:
        val_dataset_cfg = temp_cfg.val
        
    if val_dataset_cfg is None:
        logger.error("Error: Cannot find validation dataset config.")
        return

    val_dataset = instantiate_from_config(val_dataset_cfg)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    for param_key, test_values in experiments.items():
        logger.info(f"\n{'='*20} 测试参数: {param_key} {'='*20}")
        param_root_dir = os.path.join(args.save_dir, param_key)
        os.makedirs(param_root_dir, exist_ok=True)
        metric_history = {'x': [], 'PSNR': [], 'SSIM': [], 'LPIPS': []}
        
        for val in test_values:
            sub_dir_name = f"val_{val}"
            current_img_save_dir = os.path.join(param_root_dir, sub_dir_name)
            os.makedirs(current_img_save_dir, exist_ok=True)
            
            # 直接调用本地退化函数
            avg_psnr, avg_ssim, avg_lpips = run_evaluation(
                model, val_loader, loss_fn_lpips, device, 
                current_img_save_dir, param_key, val
            )
            
            logger.info(f"    [{param_key}={val}] -> PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
            metric_history['x'].append(val)
            metric_history['PSNR'].append(avg_psnr)
            metric_history['SSIM'].append(avg_ssim)
            metric_history['LPIPS'].append(avg_lpips)
        
        all_results[param_key] = metric_history

    logger.info("\n正在生成趋势图表...")
    plot_charts(all_results, args.save_dir)
    logger.info(f"全部完成！结果保存在: {args.save_dir}")

if __name__ == "__main__":
    main()
