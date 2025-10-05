# 文件: data_loader/detector_loader.py (V6.0 - 参数化解耦版)
# 职责: 作为一个纯粹的、无状态的数据集构建工具，
#       接收明确的输入/输出路径和类别定义，生成YOLOv8所需的标准数据集结构。

import sys
import json
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict
import cv2 

# --- 辅助函数 (职责单一，保持不变) ---

def _parse_labelme_json(json_path: Path, class_mapping: dict, img_h: int, img_w: int) -> List[str]:
    """解析单个LabelMe JSON文件并转换为YOLO格式字符串列表。"""
    yolo_labels = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for shape in data.get('shapes', []):
        label = shape['label'].strip()
        if label not in class_mapping:
            continue
        
        class_id = class_mapping[label]
        points = np.array(shape['points'])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        cx = (x_min + x_max) / 2 / img_w
        cy = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        
        yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return yolo_labels

def _parse_yolo_txt(txt_path: Path) -> List[str]:
    """直接读取YOLO TXT文件的内容。"""
    if not txt_path.exists():
        return []
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# --- 主构建函数 (核心重构) ---

def build_yolo_dataset(
    source_dir: Path, 
    output_dir: Path, 
    class_names: Dict[int, str], 
    val_split_ratio: float = 0.2
):
    """
    根据指定的源数据，构建一个完整的YOLOv8训练数据集。

    Args:
        source_dir (Path): 包含 .png/.jpg 和 .json/.txt 标注文件的原始数据目录。
        output_dir (Path): 用于存放生成的 train/val 数据集的目标目录。
        class_names (Dict[int, str]): 类别ID到名称的映射字典, 例如 {0: 'cat', 1: 'dog'}。
        val_split_ratio (float): 验证集所占的比例。
    """
    print(f"--- [智能数据构建] 启动 ---")
    print(f"   - 数据源: {source_dir}")
    print(f"   - 输出目录: {output_dir}")

    if not source_dir.is_dir():
        print(f"⚠️ 警告: 源数据目录 {source_dir} 不存在，跳过。")
        return

    # 【核心解耦】直接从传入的参数生成类别映射，不再依赖任何外部文件
    if not class_names:
        raise ValueError("错误: 必须提供 class_names 类别定义。")
    ID_TO_CLASS = class_names
    CLASS_TO_ID = {name: i for i, name in ID_TO_CLASS.items()}
    print(f"   - 接收到 {len(ID_TO_CLASS)} 个类别: {list(ID_TO_CLASS.values())}")

    # 准备目录结构
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images/train").mkdir(parents=True)
    (output_dir / "images/val").mkdir(parents=True)
    (output_dir / "labels/train").mkdir(parents=True)
    (output_dir / "labels/val").mkdir(parents=True)

    image_files = sorted(list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg")))
    if not image_files:
        raise FileNotFoundError(f"错误: 在 {source_dir} 中未找到任何图片文件。")

    train_files, val_files = train_test_split(image_files, test_size=val_split_ratio, random_state=42)

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        if not file_list: continue
        for image_path in tqdm(file_list, desc=f"转换 {split_name} 集"):
            shutil.copy(image_path, output_dir / f"images/{split_name}/{image_path.name}")
            
            yolo_labels = []
            txt_path = image_path.with_suffix(".txt")
            json_path = image_path.with_suffix(".json")

            if txt_path.exists():
                yolo_labels = _parse_yolo_txt(txt_path)
            elif json_path.exists():
                # [关键修复] 使用cv2动态读取每张图片的真实尺寸，杜绝硬编码
                try:
                    img = cv2.imread(str(image_path))
                    img_h, img_w, _ = img.shape
                    yolo_labels = _parse_labelme_json(json_path, CLASS_TO_ID, img_h, img_w)
                except Exception as e:
                    print(f"\n⚠️ 警告: 读取图片 {image_path.name} 尺寸失败: {e}")
                    continue # 跳过这张无法处理的图片
            
            if yolo_labels:
                label_path = output_dir / f"labels/{split_name}/{image_path.stem}.txt"
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_labels))
    
    # 创建 dataset.yaml
    dataset_yaml_data = {
        'path': str(output_dir.resolve()), 
        'train': 'images/train', 
        'val': 'images/val', 
        'names': ID_TO_CLASS
    }
    dataset_yaml_path = output_dir / "dataset.yaml"
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml_data, f, sort_keys=False, allow_unicode=True)
    
    print(f"✅ 数据集构建成功！配置文件: {dataset_yaml_path}")