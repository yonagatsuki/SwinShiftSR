import os
import argparse
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 配置区域 =================
# 固定输出目录名称
OUTPUT_DIR_512 = "ffhq_512"
OUTPUT_DIR_128 = "ffhq_128"

# 目标尺寸
SIZE_512 = (512, 512)
SIZE_128 = (128, 128)
# ===========================================

def resize_worker(file_info):
    """
    单个图片处理函数
    """
    src_path, dest_path_512, dest_path_128 = file_info
    
    try:
        with Image.open(src_path) as img:
            # 转换为 RGB 模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # --- 1. 生成并保存 512 版本 ---
            img_512 = img.resize(SIZE_512, Image.Resampling.LANCZOS)
            img_512.save(dest_path_512, quality=95)

            # --- 2. 生成并保存 128 版本 ---
            img_128 = img.resize(SIZE_128, Image.Resampling.LANCZOS)
            img_128.save(dest_path_128, quality=95)
            
            return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def main():
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="生成 ffhq_128 和 ffhq_512 数据集")
    parser.add_argument("-i", "--input", type=str, required=True, help="原始高清图片文件夹路径")
    args = parser.parse_args()

    input_dir = args.input

    # 检查输入路径
    if not os.path.exists(input_dir):
        print(f"错误：输入路径不存在 -> {input_dir}")
        return

    # --- 2. 创建固定的输出目录 ---
    for d in [OUTPUT_DIR_512, OUTPUT_DIR_128]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"已创建目录: {d}")
        else:
            print(f"目录已存在: {d}")

    # --- 3. 搜集图片 ---
    print(f"正在扫描 {input_dir} 下的所有文件...")
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))

    total_files = len(image_files)
    print(f"共找到 {total_files} 张图片。")

    if total_files == 0:
        print("错误：未找到图片。")
        return

    # --- 4. 准备任务 ---
    tasks = []
    for src_path in image_files:
        file_name = os.path.basename(src_path)
        
        # 扁平化保存到固定目录
        dest_512 = os.path.join(OUTPUT_DIR_512, file_name)
        dest_128 = os.path.join(OUTPUT_DIR_128, file_name)
        
        tasks.append((src_path, dest_512, dest_128))

    # --- 5. 多进程处理 ---
    num_processes = max(1, cpu_count() - 2)
    print(f"开始处理... (使用 {num_processes} 个进程)")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(resize_worker, tasks), total=total_files, unit="img"))

    # --- 6. 结束 ---
    success_count = sum(results)
    print(f"\n处理完成！")
    print(f"成功: {success_count} / {total_files}")
    print(f"512 数据集: {os.path.abspath(OUTPUT_DIR_512)}")
    print(f"128 数据集: {os.path.abspath(OUTPUT_DIR_128)}")

if __name__ == '__main__':
    main()
