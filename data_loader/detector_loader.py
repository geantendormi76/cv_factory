# 文件: data_loader/detector_loader.py (V5.0 - 智能多格式版)
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.constants import get_class_maps_from_yolo_config

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

def build_yolo_dataset(config: Dict):
    data_paths = config['data_paths']
    # 【核心修改】从新的统一任务配置中获取路径
    task_name = config['current_task']
    source_dir = Path(data_paths[f'{task_name}_source_dir'])
    output_dir = Path(data_paths[f'{task_name}_output_dir'])
    val_split_ratio = 0.2

    print(f"--- [智能数据构建] 开始处理统一源: {source_dir} ---")
    
    if not source_dir.is_dir():
        print(f"⚠️ 警告: 源数据目录 {source_dir} 不存在，跳过。")
        return

    task_config = config['tasks'][task_name]
    yolo_config_path = PROJECT_ROOT / 'configs' / task_config['yolo_config_path']
    
    try:
        CLASS_TO_ID, ID_TO_CLASS = get_class_maps_from_yolo_config(yolo_config_path)
    except Exception as e:
        print(f"❌ 错误: 加载类别信息失败: {e}")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images/train").mkdir(parents=True); (output_dir / "images/val").mkdir(parents=True)
    (output_dir / "labels/train").mkdir(parents=True); (output_dir / "labels/val").mkdir(parents=True)

    # 【核心修改】现在我们遍历图片文件，而不是JSON文件
    image_files = sorted(list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg")))
    if not image_files:
        print(f"❌ 错误: 在 {source_dir} 中未找到任何图片文件。")
        return

    train_files, val_files = train_test_split(image_files, test_size=val_split_ratio, random_state=42)

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        if not file_list: continue
        for image_path in tqdm(file_list, desc=f"转换 {split_name} 集"):
            # 1. 复制图片
            shutil.copy(image_path, output_dir / f"images/{split_name}/{image_path.name}")
            
            # 2. 【智能查找】优先找.txt，再找.json
            yolo_labels = []
            txt_path = image_path.with_suffix(".txt")
            json_path = image_path.with_suffix(".json")

            if txt_path.exists():
                yolo_labels = _parse_yolo_txt(txt_path)
            elif json_path.exists():
                # JSON解析需要图像尺寸
                img_h, img_w = 600, 800 # 假设固定尺寸，更健壮的做法是用cv2读取
                yolo_labels = _parse_labelme_json(json_path, CLASS_TO_ID, img_h, img_w)
            
            # 3. 写入标签
            if yolo_labels:
                label_path = output_dir / f"labels/{split_name}/{image_path.stem}.txt"
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(yolo_labels))
                
    dataset_yaml_data = {'path': str(output_dir.resolve()), 'train': 'images/train', 'val': 'images/val', 'names': ID_TO_CLASS}
    dataset_yaml_path = output_dir / "dataset.yaml"
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml_data, f, sort_keys=False, allow_unicode=True)
    
    print(f"✅ [智能数据构建] 统一数据集构建成功！\n   输出目录: {output_dir}")