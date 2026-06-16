import os
import cv2
import torch
import numpy as np
import argparse
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from omegaconf import OmegaConf

from stage1_utils.swinir import SwinIR
from stage1_utils.codeformer import CodeformerDataset

# ================= 配置区域 (请确认这里) =================
CKPT_PATH = "weights/swinir.pt"
CONFIG_PATH = "configs/train_stage1.yaml"

# 定义基础输出目录
BASE_CLEAN_DIR = "data/FFHQ_SwinIR_128_Clean"
BASE_LQ_DIR = "data/FFHQ_128_LQ_Generated"

# 定义两个任务：(任务名称, 列表文件路径, 配置文件中的key, Clean保存路径, LQ保存路径)
TASKS = [
    {
        "name": "Train",
        "file_list": "files_shuf_train.list",
        "config_key": "train",
        "clean_save_dir": os.path.join(BASE_CLEAN_DIR, "train"),
        "lq_save_dir": os.path.join(BASE_LQ_DIR, "train") # 新增：LQ保存路径
    },
    {
        "name": "Val",
        "file_list": "files_shuf_val.list",
        "config_key": "val",
        "clean_save_dir": os.path.join(BASE_CLEAN_DIR, "val"),
        "lq_save_dir": os.path.join(BASE_LQ_DIR, "val") # 新增：LQ保存路径
    }
]
# =======================================================

def tensor_to_img_np(tensor):
    """辅助函数：将 (C,H,W) Tensor 转换为 OpenCV 可保存的 numpy 数组"""
    # tensor 在 CPU 上，形状 (C, H, W)，范围 0-1
    img_np = tensor.clamp(0, 1).numpy().transpose(1, 2, 0)
    # 0-1 -> 0-255 uint8
    img_np = (img_np * 255.0).round().astype(np.uint8)
    # RGB -> BGR
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np

def process_split(task, gpu_id, rank, world_size, model, cfg, device):
    """
    处理单个数据集切片的核心函数
    """
    split_name = task["name"]
    file_list_path = task["file_list"]
    config_key = task["config_key"]
    clean_save_dir = task["clean_save_dir"]
    lq_save_dir = task["lq_save_dir"]

    # 1. 创建保存目录 (Clean 和 LQ)
    os.makedirs(clean_save_dir, exist_ok=True)
    os.makedirs(lq_save_dir, exist_ok=True)

    # 2. 获取对应的 Dataset 参数
    dataset_params = cfg.dataset[config_key].params
    # 【强制覆盖】确保关键参数正确
    dataset_params.out_size = 128
    dataset_params.file_list = file_list_path

    # 3. 初始化 Dataset
    # 这里会根据 yaml 里 train/val 不同的模糊参数来在线生成 LQ 数据
    dataset = CodeformerDataset(**dataset_params)
    
    # 4. 数据切分 (Multi-GPU Sharding)
    indices = list(range(rank, len(dataset), world_size))
    if len(indices) == 0:
        print(f" [GPU {gpu_id}] {split_name} 数据集分到的任务为空")
        return

    subset = Subset(dataset, indices)
    
    # 5. DataLoader
    # batch_size 可以设大一点 (例如 128) 加速，因为只是推理和保存
    dataloader = DataLoader(subset, batch_size=72, shuffle=False, num_workers=4, drop_last=False)

    # 6. 读取文件名列表 (用于保存)
    with open(file_list_path, 'r') as f:
        all_file_paths = [line.strip() for line in f.readlines()]
    
    # 切分文件名列表 (必须与 Subset 逻辑一致)
    my_file_paths = all_file_paths[rank::world_size]

    # 7. 开始处理循环
    global_idx = 0
    
    # 进度条描述
    desc_str = f"[GPU {gpu_id}] Processing {split_name}"
    iterator = tqdm(dataloader, desc=desc_str, position=gpu_id, leave=False)
    
    with torch.no_grad():
        for batch in iterator:
            # batch: (gt, lq_cpu, prompt)
            # lq_cpu 是 DataLoader 生成出来的烂图 Tensor，还在 CPU 上
            lq_cpu = batch[1]

            # 确保维度是 (B, C, H, W)
            if lq_cpu.shape[-1] == 3: 
                lq_cpu = lq_cpu.permute(0, 3, 1, 2)
            
            # --- A. 将 LQ 送入 GPU 进行 SwinIR 修复 ---
            lq_gpu = lq_cpu.to(device)
            output_gpu = model(lq_gpu)

            # --- B. 循环保存 LQ 和 Clean 图片 ---
            # 我们使用 CPU 上的 lq_cpu 来保存烂图，使用 GPU 算出来的 output_gpu 来保存修复图
            for i in range(output_gpu.shape[0]):
                if global_idx >= len(my_file_paths):
                    break
                
                original_path = my_file_paths[global_idx]
                file_name = os.path.basename(original_path)
                
                # 1. 保存 LQ (烂图)
                # 取出单个样本，保持在 CPU
                img_tensor_lq = lq_cpu[i] 
                img_np_lq = tensor_to_img_np(img_tensor_lq)
                cv2.imwrite(os.path.join(lq_save_dir, file_name), img_np_lq)

                # 2. 保存 Clean (SwinIR 修复图)
                # 取出单个样本，转到 CPU
                img_tensor_clean = output_gpu[i].cpu()
                img_np_clean = tensor_to_img_np(img_tensor_clean)
                cv2.imwrite(os.path.join(clean_save_dir, file_name), img_np_clean)
                
                global_idx += 1
    
    print(f" [GPU {gpu_id}] {split_name} 完成！({len(indices)} 张)")


def worker_process(gpu_id, gpu_list):
    """
    进程主函数
    """
    rank = gpu_list.index(gpu_id)
    world_size = len(gpu_list)
    device = f"cuda:{gpu_id}"
    
    print(f"🚀 [GPU {gpu_id}] 进程启动...")

    # 1. 加载配置
    cfg = OmegaConf.load(CONFIG_PATH)

    # 2. 加载模型
    print(f"[GPU {gpu_id}] 加载 SwinIR 模型...")
    model_params = cfg.model.swinir.params
    model_params.sf = 2
    model_params.unshuffle_scale = 2
    model_params.img_size = 64
    
    model = SwinIR(**model_params)
    
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"权重文件未找到: {CKPT_PATH}")

    sd = torch.load(CKPT_PATH, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    
    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    model.eval()

    # 3. 依次处理每个任务 (Train 和 Val)
    for task in TASKS:
        process_split(task, gpu_id, rank, world_size, model, cfg, device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0", help="指定GPU列表，例如 0,1,2")
    args = parser.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(',')]
    
    print(f" 准备在以下显卡上并行处理: {gpu_list}")
    print("任务输出目录:")
    for t in TASKS:
        print(f"  - {t['name']} Clean: {t['clean_save_dir']}")
        print(f"  - {t['name']} LQ:    {t['lq_save_dir']}")

    mp.set_start_method('spawn', force=True)
    processes = []
    
    for gpu_id in gpu_list:
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_list))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("\n 所有任务处理完毕！")

if __name__ == "__main__":
    main()
