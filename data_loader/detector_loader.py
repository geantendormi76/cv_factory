# 文件: data_loader/detector_loader.py (V4.0 - 鲁棒性清理最终版)
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict
import yaml

# 确保能导入utils模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.constants import get_class_maps_from_yolo_config

def build_yolo_dataset(config: Dict):
    # ... (此函数的上半部分完全不变，从 data_paths 到 train_test_split) ...
    data_paths = config['data_paths']
    source_dir = Path(data_paths['detector_source_dir'])
    output_dir = Path(data_paths['detector_output_dir'])
    val_split_ratio = 0.2

    print(f"--- [数据构建] 开始处理源数据: {source_dir} ---")
    
    if not source_dir.is_dir():
        print(f"⚠️ 警告: 源数据目录 {source_dir} 不存在，跳过数据准备。")
        return

    task_config = config['tasks']['detector']
    yolo_config_filename = task_config['yolo_config_path']
    yolo_config_path = PROJECT_ROOT / 'configs' / yolo_config_filename
    
    try:
        CLASS_TO_ID, ID_TO_CLASS = get_class_maps_from_yolo_config(yolo_config_path)
    except Exception as e:
        print(f"❌ 错误: 加载类别信息失败。详情: {e}")
        return

    if output_dir.exists():
        print(f"清理旧数据集目录: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True); (output_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/train").mkdir(parents=True, exist_ok=True); (output_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    json_files = list(source_dir.glob("*.json"))
    if not json_files:
        print(f"❌ 错误: 在 {source_dir} 中未找到任何 .json 文件。")
        return

    train_files, val_files = train_test_split(json_files, test_size=val_split_ratio, random_state=42)
    print(f"数据集划分完成: {len(train_files)} 训练, {len(val_files)} 验证。")

    # 【鲁棒性增强】增加一个集合，用于跟踪所有未匹配的标签，避免重复打印警告
    unmatched_labels_tracker = set()

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        if not file_list: continue
        for json_path in tqdm(file_list, desc=f"转换 {split_name} 集"):
            image_path = json_path.with_suffix(".png")
            if not image_path.exists(): continue
            
            shutil.copy(image_path, output_dir / f"images/{split_name}/{image_path.name}")
            
            yolo_labels = convert_single_json_to_yolo(json_path, CLASS_TO_ID, unmatched_labels_tracker)
            label_path = output_dir / f"labels/{split_name}/{json_path.stem}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_labels))

    if unmatched_labels_tracker:
        print("\n" + "="*50)
        print("⚠️ [数据构建警告] 检测到以下标签无法匹配，已被跳过:")
        for lbl in sorted(list(unmatched_labels_tracker)):
            print(f"  - '{lbl}'")
        print("="*50 + "\n")
                
    dataset_yaml_data = {'path': str(output_dir.resolve()), 'train': 'images/train', 'val': 'images/val', 'names': ID_TO_CLASS}
    dataset_yaml_path = output_dir / "dataset.yaml"
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml_data, f, sort_keys=False, allow_unicode=True)
    
    print(f"✅ [数据构建] 检测数据集构建成功！\n   输出目录: {output_dir}")

def convert_single_json_to_yolo(json_path: Path, class_mapping: dict, unmatched_tracker: set) -> List[str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_h, img_w = data['imageHeight'], data['imageWidth']
    yolo_labels = []
    for shape in data['shapes']:
        # 【核心修正】在进行任何比较之前，对标签进行清理！
        # .strip() 会移除所有前导和尾随的空白字符（空格、制表符、换行符等）。
        # 这是数据处理的黄金法则。
        label = shape['label'].strip()
            
        if label not in class_mapping:
            if label not in unmatched_tracker:
                unmatched_tracker.add(label) # 记录未匹配的标签
            continue
        
        class_id = class_mapping[label]
        points = np.array(shape['points'])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        cx = (x_min + x_max) / 2 / img_w; cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w; h = (y_max - y_min) / img_h
        
        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    return yolo_labels