import os
import argparse
from argparse import ArgumentParser
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import save_image
import lpips

from stage1_utils.swinir import SwinIR
from stage1_utils.common import instantiate_from_config, calculate_psnr_pt

def set_seed(seed=42):
    """
    Lock all possible random sources to ensure results are reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure CuDNN uses deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Random Seed set to: {seed}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument("--ckpt", type=str, required=True, help="Model weight path (.pt)")
    parser.add_argument("--save_dir", type=str, default=None, help="Image save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default 42")
    args = parser.parse_args()

    # 1. Lock seed first!
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 2. Load config
    cfg = OmegaConf.load(args.config)

    # 3. Initialize model
    print(f"[INFO] Loading model structure...")
    model = instantiate_from_config(cfg.model.swinir)
    
    # 4. Load weights (Handle DDP module. prefix)
    print(f"[INFO] Loading weights: {os.path.basename(args.ckpt)}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    
    # Smart removal of 'module.' prefix
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()

    # 5. Prepare validation data
    print("[INFO] Loading validation set...")
    val_dataset = instantiate_from_config(cfg.dataset.val)
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        drop_last=False
    )
    
    # 6. Prepare LPIPS evaluation model
    print("[INFO] Initializing LPIPS metric...")
    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    loss_fn_lpips.eval()

    # 7. Start testing loop
    total_psnr = 0.0
    total_lpips = 0.0
    total_mse = 0.0 
    count = 0
    
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[INFO] Result images will be saved to: {args.save_dir}")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    print(f"[INFO] Starting test on {len(val_loader)} images...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Testing")):
            # Data transfer
            if isinstance(batch, (list, tuple)):
                 batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, dict):
                 batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            try:
                batch = batch_transform(batch)
            except:
                pass

            # Parse data
            if isinstance(batch, dict):
                lq = batch.get('lq')
                gt = batch.get('gt')
                if lq is None and 'image' in batch:
                     lq = batch['lq']
                     gt = batch['gt']
            else:
                gt, lq = batch[0], batch[1]

            if gt.ndim == 5: 
                gt = gt.squeeze(0)
                lq = lq.squeeze(0)

            # === Data Normalization Logic ===
            # Force data to [0, 1] range
            if gt.min() < 0: 
                gt = (gt + 1.0) / 2.0
            if lq.min() < 0: 
                lq = (lq + 1.0) / 2.0
            
            # Clamp again for safety
            gt = torch.clamp(gt, 0, 1)
            lq = torch.clamp(lq, 0, 1)
            
            # Dimension adjustment
            if gt.shape[1] != 3 and gt.shape[3] == 3: 
                gt = rearrange(gt, "b h w c -> b c h w")
                lq = rearrange(lq, "b h w c -> b c h w")
                
            gt = gt.contiguous().float()
            lq = lq.contiguous().float()

            # === Inference ===
            pred = model(lq)
            pred = torch.clamp(pred, 0, 1)

            # === Calculate Metrics ===
            # 1. PSNR
            psnr = calculate_psnr_pt(pred, gt, crop_border=0).mean().item()
            
            # 2. LPIPS
            lpips_val = loss_fn_lpips(pred, gt, normalize=True).mean().item()
            
            # 3. MSE Loss
            mse_val = F.mse_loss(pred, gt, reduction='mean').item()

            total_psnr += psnr
            total_lpips += lpips_val
            total_mse += mse_val
            count += 1

            # === Save Image (Optional) ===
            if args.save_dir:
                save_name = f"{i:05d}_P{psnr:.2f}_L{lpips_val:.3f}.png"
                lq_resized = torch.nn.functional.interpolate(lq, size=gt.shape[-2:], mode='nearest')
                comparison = torch.cat([lq_resized, pred, gt], dim=3) 
                save_image(comparison, os.path.join(args.save_dir, save_name))

    # 8. Output Final Results
    avg_psnr = total_psnr / count
    avg_lpips = total_lpips / count
    avg_mse = total_mse / count

    print("\n" + "="*60)
    print(f"TEST REPORT")
    print(f"------------------------------------------------------------")
    print(f"Model Path   : {os.path.basename(args.ckpt)}")
    print(f"Random Seed  : {args.seed}")
    print(f"Test Count   : {count} images")
    print(f"------------------------------------------------------------")
    print(f"Average PSNR : {avg_psnr:.4f} (Higher is Better)")
    print(f"Average LPIPS: {avg_lpips:.4f} (Lower is Better)")
    print(f"Average MSE  : {avg_mse:.6f} (Lower is Better)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
