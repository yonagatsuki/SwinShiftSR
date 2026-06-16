import os
import argparse
import random
import numpy as np
import torch
import cv2
import logging
import sys
import math
import importlib
from tqdm import tqdm
from torchvision.utils import make_grid
from omegaconf import OmegaConf
import torch.nn.functional as F
import torch.nn as nn


def switch_to_project_root():
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, "../SwinIR_ResShift"))
    
    if os.path.exists(project_root) and os.path.exists(os.path.join(project_root, "stage1_utils")):
        print(f"[INFO] Switching working directory to: {project_root}")
        if project_root not in sys.path:
            sys.path.append(project_root)
        return project_root
    
    if os.path.exists(os.path.join(current_dir, "stage1_utils")):
        return current_dir

    print(f"[WARN] Could not find project root containing 'stage1_utils'. Using: {current_dir}")
    return current_dir

PROJECT_ROOT = switch_to_project_root()

try:
    import albumentations
except ImportError:
    import types
    sys.modules['albumentations'] = types.ModuleType('albumentations')
    print("[INFO] Mocked 'albumentations' module for inference.")

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    try:
        import torchvision.transforms.functional
        sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional
    except ImportError:
        pass

# === LPIPS Import ===
try:
    import lpips
except ImportError as e:
    print(f"!!! Error: lpips not found: {e}")
    print("!!! Please run: pip install lpips")
    exit()


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class LocalRealESRGANer:
    def __init__(self, scale, model_path, model, device=torch.device('cuda'), pre_pad=10, tile=0, tile_pad=10, half=False):
        self.scale = scale
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.pre_pad = pre_pad
        self.tile = tile
        self.tile_pad = tile_pad
        self.half = False 
        
        if os.path.exists(model_path):
            print(f"[INFO] Loading weights from {model_path}...")
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            elif 'params' in loadnet:
                keyname = 'params'
            else:
                keyname = None
            
            state_dict = loadnet[keyname] if keyname else loadnet
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=True)
            self.model.float()
        else:
            print(f"[WARN] Model path {model_path} does not exist.")

    def enhance(self, img, outscale=None):
        img = img.astype(np.float32) / 255.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        if self.pre_pad != 0:
            img = F.pad(img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        
        with torch.no_grad():
            try:
                output = self.model(img)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("[WARN] OOM Error in RealESRGAN inference!")
                raise e
            
        if self.pre_pad != 0:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output, None

# ================= 配置区域 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_TO_SAVE = 20     
MAX_TEST_SAMPLES = None 
# ===========================================

def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "test_log.txt")
    logger = logging.getLogger("RealESRGAN_Test")
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

# === [Metrics] 内置计算 ===
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
    mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
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

# === 退化逻辑 ===
def apply_degradation(img_gt, params):
    img_gt = img_gt.astype(np.float32)
    h, w = img_gt.shape[:2]
    img_lq = cv2.resize(img_gt, (128, 128), interpolation=cv2.INTER_LINEAR)
    h_lq, w_lq = 128, 128
    sigma = np.random.uniform(params['blur_sigma'][0], params['blur_sigma'][1])
    if sigma > 0.1:
        k_size = int(sigma * 4) + 1
        if k_size % 2 == 0: k_size += 1
        k_size = max(3, k_size)
        img_lq = cv2.GaussianBlur(img_lq, (k_size, k_size), sigma)
    scale = np.random.uniform(params['downsample_range'][0], params['downsample_range'][1])
    if scale > 1.0:
        img_lq = cv2.resize(img_lq, (int(w_lq // scale), int(h_lq // scale)), interpolation=cv2.INTER_LINEAR)
        img_lq = cv2.resize(img_lq, (w_lq, h_lq), interpolation=cv2.INTER_LINEAR)
    noise_level = np.random.uniform(params['noise_range'][0], params['noise_range'][1])
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img_lq.shape)
        img_lq = img_lq + noise
    jpeg_q = random.randint(params['jpeg_range'][0], params['jpeg_range'][1])
    if jpeg_q < 100:
        img_lq = np.clip(img_lq, 0, 255).astype(np.uint8)
        _, encimg = cv2.imencode('.jpg', img_lq, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        img_lq = cv2.imdecode(encimg, 1)
        img_lq = img_lq.astype(np.float32)
    if img_lq.shape[:2] != (128, 128):
        img_lq = cv2.resize(img_lq, (128, 128), interpolation=cv2.INTER_LINEAR)
    return np.clip(img_lq, 0, 255).astype(np.uint8)

# === 模型加载 ===
def load_realesrgan(model_path):
    print(f"[INFO] Loading Real-ESRGAN from {model_path}...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = LocalRealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=10, 
        half=False, 
        device=torch.device(DEVICE)
    )
    return upsampler

# [New] 安全保存函数
def save_tensor_img(tensor, path):
    """
    使用 OpenCV 保存 Tensor 图片，绕过 PIL/Torchvision 的 bug
    tensor: (C, H, W) RGB [0, 1]
    """
    # 1. Tensor -> Numpy (H, W, C) [0, 255] RGB
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # 2. RGB -> BGR
    im_bgr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    # 3. Save
    cv2.imwrite(path, im_bgr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_list", type=str, required=True, help="Path to validation list")
    parser.add_argument("--data_root", type=str, default="", help="Root prefix for images")
    parser.add_argument("--ckpt", type=str, default="weights/RealESRGAN_x4plus.pth", help="Model checkpoint")
    parser.add_argument("--save_dir", type=str, default="results_realesrgan", help="Output directory")
    args = parser.parse_args()

    args.file_list = os.path.abspath(args.file_list)
    args.data_root = os.path.abspath(args.data_root)
    args.ckpt = os.path.abspath(args.ckpt)
    args.save_dir = os.path.abspath(args.save_dir)

    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
        print(f"[INFO] Current Working Directory changed to: {os.getcwd()}")

    # [Fix] 智能路径修正逻辑
    if not os.path.exists(args.file_list):
        # 移除可能重复的前缀 (SwinIR_ResShift/SwinIR_ResShift/...)
        potential_path = os.path.join(PROJECT_ROOT, os.path.basename(args.file_list))
        if os.path.exists(potential_path):
            print(f"[INFO] Auto-corrected file_list path to: {potential_path}")
            args.file_list = potential_path

    set_seed(123456)
    logger = setup_logger(args.save_dir)
    
    logger.info(f"Loading file list: {args.file_list}")
    if not os.path.exists(args.file_list):
        logger.error(f"File list not found: {args.file_list}")
        return

    with open(args.file_list, 'r') as f:
        img_paths = [line.strip() for line in f.readlines()]
    logger.info(f"Total images: {len(img_paths)}")

    if not os.path.exists(args.ckpt):
        logger.error(f"Checkpoint not found: {args.ckpt}")
        return
    upsampler = load_realesrgan(args.ckpt)
    
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(DEVICE).eval()

    degradation_params = {
        'blur_sigma': [0.1, 3.0],
        'downsample_range': [1, 3],
        'noise_range': [0, 15],
        'jpeg_range': [30, 100]
    }

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    count = 0
    saved_count = 0
    
    logger.info("Starting Inference...")
    
    for i, img_path in enumerate(tqdm(img_paths)):
        if MAX_TEST_SAMPLES and count >= MAX_TEST_SAMPLES: break
        
        # 确保 img_path 没有重复的前缀
        if args.data_root.endswith(os.path.basename(args.data_root)) and img_path.startswith(os.path.basename(args.data_root)):
             # 如果 data_root 是 .../A，img_path 是 A/file.jpg，则去掉 img_path 的 A/
             img_path = img_path[len(os.path.basename(args.data_root))+1:]

        # 尝试不同组合
        candidates = [
            os.path.join(args.data_root, img_path),
            os.path.abspath(img_path)
        ]
        
        img_gt = None
        for p in candidates:
            if os.path.exists(p):
                img_gt = cv2.imread(p)
                break
        
        if img_gt is None:
            # 静默跳过，避免刷屏
            continue
            
        img_gt = cv2.resize(img_gt, (512, 512), interpolation=cv2.INTER_CUBIC)
        img_lq = apply_degradation(img_gt, degradation_params)
        
        try:
            output_bgr, _ = upsampler.enhance(img_lq, outscale=4)
        except Exception as e:
            logger.error(f"Inference failed at {i}: {e}")
            continue

        gt_tensor = torch.from_numpy(img_gt[:, :, ::-1].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        gt_tensor = gt_tensor.to(DEVICE)
        
        pred_tensor = torch.from_numpy(output_bgr[:, :, ::-1].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        pred_tensor = pred_tensor.to(DEVICE)
        
        lq_tensor = torch.from_numpy(img_lq[:, :, ::-1].copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        lq_tensor = lq_tensor.to(DEVICE)

        p = calculate_psnr_pt(pred_tensor, gt_tensor).mean().item()
        l = loss_fn_lpips(pred_tensor, gt_tensor, normalize=True).mean().item()
        s = calculate_ssim_pt(pred_tensor, gt_tensor).item()

        total_psnr += p
        total_lpips += l
        total_ssim += s
        count += 1
        
        if saved_count < IMAGES_TO_SAVE:
            lq_viz = F.interpolate(lq_tensor, size=(512, 512), mode='nearest')
            grid = make_grid([gt_tensor[0], lq_viz[0], pred_tensor[0]], nrow=3, padding=0)
            
            save_path = os.path.join(args.save_dir, f"realesrgan_{i:04d}_P{p:.2f}_S{s:.3f}_L{l:.3f}.png")
            save_tensor_img(grid, save_path)
            saved_count += 1

    if count > 0:
        logger.info("="*40)
        logger.info(f"Real-ESRGAN Test Results ({count} samples)")
        logger.info("="*40)
        logger.info(f"PSNR : {total_psnr / count:.4f}")
        logger.info(f"SSIM : {total_ssim / count:.4f}")
        logger.info(f"LPIPS: {total_lpips / count:.4f}")
        logger.info("="*40)

if __name__ == "__main__":
    main()