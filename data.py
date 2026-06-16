import os
import cv2
import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys

# ==============================================================================
# [Path Setup] 自动定位项目根目录以便导入 stage1_utils
# ==============================================================================
def setup_paths(swinir_root):
    swinir_abs = os.path.abspath(swinir_root)
    if os.path.exists(swinir_abs):
        if swinir_abs not in sys.path:
            sys.path.append(swinir_abs)
            print(f"[INFO] Added SwinIR root to sys.path: {swinir_abs}")
    else:
        print(f"[ERROR] SwinIR root not found at {swinir_abs}")
        exit()

def set_seed(seed=123456):
    """
    完全复刻参考代码的种子设置逻辑
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="通过复刻 Dataset 流程生成完全一致的固化测试集")
    parser.add_argument("--config", type=str, required=True, help="SwinIR_ResShift 配置文件路径 (.yaml)")
    parser.add_argument("--swinir_root", type=str, default=".", help="SwinIR 项目根目录")
    parser.add_argument("--save_dir", type=str, default="fixed_val_set", help="结果保存目录")
    parser.add_argument("--seed", type=int, default=123456, help="随机种子 (需与参考代码一致)")
    parser.add_argument("--limit", type=int, default=None, help="限制生成的数量")
    args = parser.parse_args()

    setup_paths(args.swinir_root)
    
    # 动态导入项目内的实例化工具
    try:
        from stage1_utils.common import instantiate_from_config
    except ImportError:
        print("[ERROR] 无法导入 stage1_utils。请确保 --swinir_root 指向正确的目录。")
        exit()

    # 1. 加载配置并准备 Dataset
    cfg = OmegaConf.load(args.config)
    
    # 自动寻找验证集配置
    val_dataset_cfg = None
    if 'data' in cfg and 'val' in cfg.data: val_dataset_cfg = cfg.data.val
    elif 'dataset' in cfg and 'val' in cfg.dataset: val_dataset_cfg = cfg.dataset.val
    
    if val_dataset_cfg is None:
        print("[ERROR] 配置文件中未找到验证集配置 (data.val 或 dataset.val)")
        return

    # 2. 准备目录
    save_root = Path(args.save_dir)
    gt_dir = save_root / "gt"
    lq_dir = save_root / "lq"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)

    # 3. 实例化 Dataset 和 DataLoader (必须 num_workers=0 才能保证顺序随机性一致)
    val_dataset = instantiate_from_config(val_dataset_cfg)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 4. 执行双重种子设置 (复刻参考代码逻辑)
    set_seed(args.seed)
    
    print(f"[INFO] 正在启动生成流程。种子: {args.seed}")
    print(f"[INFO] 样本总数: {len(val_dataset)}")
    
    # 在推理/循环前再次重置种子 (完全复刻参考代码的操作)
    set_seed(args.seed)

    success_count = 0
    
    for i, batch in enumerate(tqdm(val_loader)):
        if args.limit and i >= args.limit:
            break
        
        # 解析 Dataset 返回的内容
        if isinstance(batch, dict):
            # 针对 ResshiftOnlineDataset
            gt_tensor = batch['gt']
            lq_tensor = batch['lq']
        else:
            # 针对 CodeformerDataset
            gt_tensor, lq_tensor = batch[0], batch[1]

        # 去掉 Batch 维度
        gt_tensor = gt_tensor.squeeze(0)
        lq_tensor = lq_tensor.squeeze(0)

        # 转换回 [0, 255] Numpy BGR 以便保存
        # GT: [-1, 1] -> [0, 255]
        img_gt = ((gt_tensor.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)

        # LQ: [0, 1] -> [0, 255]
        img_lq = (lq_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)

        # 保存文件名：固定序号
        base_name = f"{i:05d}.png"

        cv2.imwrite(str(gt_dir / base_name), img_gt)
        cv2.imwrite(str(lq_dir / base_name), img_lq)
        success_count += 1

    print(f"\n" + "="*50)
    print(f"[Summary] 处理完成！")
    print(f" - 成功固化: {success_count} 对图像")
    print(f" - 存储位置: {args.save_dir}")
    print(f"提示：该脚本直接调用了您的 Dataset 类，生成的 LQ 图像应与测试代码完全一致。")
    print("="*50)

if __name__ == "__main__":
    main()