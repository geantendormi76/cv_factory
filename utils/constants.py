# 文件: /home/zhz/deepl/utils/constants.py (V7.0 - 动态YAML加载器最终版)
# 职责: 提供一个健壮的工具，用于从YOLOv8训练配置文件(.yaml)中动态加载类别信息。
#       本模块不再是静态的“事实来源”，而是一个动态的、按需服务的“信息解析器”。

import yaml
from pathlib import Path
from typing import Dict, Tuple, List

def get_class_maps_from_yolo_config(yolo_config_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    从指定的YOLO配置文件中读取'names'字段，并生成类别到ID和ID到类别的映射。

    该函数具有高鲁棒性，能够智能处理两种常见的 'names' 格式:
    1. 字典格式 (YOLOv8 推荐): {0: 'cat', 1: 'dog'}
    2. 列表格式 (旧版YOLOv5兼容): ['cat', 'dog']

    Args:
        yolo_config_path (Path): 指向YOLO训练配置 .yaml 文件的路径对象。

    Returns:
        A tuple containing:
        - class_to_id (Dict[str, int]): A dictionary mapping class names to IDs.
        - id_to_class (Dict[int, str]): A dictionary mapping IDs to class names.
        
    Raises:
        FileNotFoundError: 如果配置文件不存在。
        KeyError: 如果配置文件中缺少 'names' 字段。
        ValueError: 如果 'names' 字段为空或格式不正确。
    """
    print(f"--- [类别加载] 动态加载类别来源: {yolo_config_path.name} ---")
    if not yolo_config_path.is_file():
        raise FileNotFoundError(f"❌ 错误: 动态类别源文件未找到: {yolo_config_path}")

    with open(yolo_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'names' not in config:
        raise KeyError(f"❌ 错误: 在 {yolo_config_path.name} 中未找到关键的 'names' 字段。")

    names_data = config['names']
    if not names_data:
        raise ValueError(f"❌ 错误: 'names' 字段在 {yolo_config_path.name} 中为空。")

    # --- 核心逻辑: 智能解析 'names' ---
    if isinstance(names_data, dict):
        # YOLOv8 格式: {0: 'name1', 1: 'name2'}
        id_to_class = dict(sorted(names_data.items())) # 确保ID有序
        class_to_id = {name: i for i, name in id_to_class.items()}
    elif isinstance(names_data, list):
        # YOLOv5 兼容格式: ['name1', 'name2']
        id_to_class = {i: name for i, name in enumerate(names_data)}
        class_to_id = {name: i for i, name in id_to_class.items()}
    else:
        raise TypeError(f"❌ 错误: 'names' 字段的格式无法识别。期望是字典或列表，但得到的是 {type(names_data)}。")

    print(f"✅ [类别加载] 成功加载 {len(id_to_class)} 个类别。")
    return class_to_id, id_to_class