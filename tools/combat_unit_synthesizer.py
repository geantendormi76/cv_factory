# 文件名: tools/combat_unit_synthesizer.py (V2.0 - 流水线直通版)
# 职责: 生成作战单位数据，并直接输出为 pipeline.py 所期望的“原始数据”格式。

import cv2
import numpy as np
import yaml
from pathlib import Path
import random
import shutil
from tqdm import tqdm
from itertools import cycle

# ==================================================================
# --- 1. 核心配置区 ---
# ==================================================================
CONFIG = {
    # --- 输入路径 ---
    "FOREGROUNDS_DIR": Path("/home/zhz/deepl/data/assets/foregrounds"),
    "BACKGROUNDS_DIR": Path("/home/zhz/deepl/data/assets/backgrounds/combat"),
    
    # --- 【核心修改】输出路径直接指向 "raw" 目录 ---
    "OUTPUT_DIR": Path("/home/zhz/deepl/data/raw/synthetic_combat_units_v1_raw"),

    # --- 生成参数 ---
    "NUM_IMAGES_TO_GENERATE": 10000,
    
    # --- 场景约束 ---
    "MIN_UNITS_PER_IMAGE": 4,
    "MAX_UNITS_PER_IMAGE": 20,
    "OVERLAP_THRESHOLD_IOU": 0.3,
    "MAX_PLACEMENT_ATTEMPTS": 50,
}
# ==================================================================


# ... (辅助函数 calculate_iou 和 paste_transparent 保持不变) ...
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0: return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def paste_transparent(background, foreground, position):
    if foreground.shape[2] < 4: return background
    x, y = position
    h, w = foreground.shape[:2]
    x_end, y_end = x + w, y + h
    if x_end > background.shape[1] or y_end > background.shape[0] or x < 0 or y < 0: return background
    alpha = foreground[:, :, 3] / 255.0
    alpha_mask = np.dstack([alpha] * 3)
    bg_roi = background[y:y_end, x:x_end, :3]
    blended_roi = (foreground[:, :, :3] * alpha_mask) + (bg_roi * (1 - alpha_mask))
    background[y:y_end, x:x_end, :3] = blended_roi.astype(background.dtype)
    return background


class CombatUnitSynthesizer:
    def __init__(self, config):
        self.config = config
        self.fg_dir = self.config["FOREGROUNDS_DIR"]
        self.bg_dir = self.config["BACKGROUNDS_DIR"]
        self.output_dir = self.config["OUTPUT_DIR"]
        
        self.assets = {}
        self.backgrounds = []
        self.class_names = []
        self.class_to_id = {}
        
        self.bg_cycler = None
        self.class_cycler = None

    def _prepare_environment(self):
        """【核心修改】准备一个单一的 "raw" 输出目录"""
        print("--- [1/3] 准备环境中... ---")
        if self.output_dir.exists():
            print(f"警告: 输出目录 {self.output_dir} 已存在，将进行覆盖。")
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"数据将直接生成到: {self.output_dir}")

    def _load_assets(self):
        """加载所有前景和背景资产"""
        print("--- [2/3] 加载资产库... ---")
        bg_files = list(self.bg_dir.glob("*.png")) + list(self.bg_dir.glob("*.jpg"))
        for path in tqdm(bg_files, desc="加载背景"):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None: self.backgrounds.append(img)
        
        if not self.backgrounds: raise FileNotFoundError(f"错误: 在 {self.bg_dir} 中未找到任何背景图片。")
        self.bg_img_h, self.bg_img_w = self.backgrounds[0].shape[:2]
        
        class_dirs = sorted([d for d in self.fg_dir.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}

        for class_dir in tqdm(class_dirs, desc="加载前景资产"):
            class_name = class_dir.name
            self.assets[class_name] = []
            for asset_path in class_dir.glob("*.png"):
                asset_img = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
                if asset_img is not None and asset_img.shape[2] == 4:
                    self.assets[class_name].append(asset_img)
        
        print(f"✅ 成功加载 {len(self.backgrounds)} 张背景和 {len(self.class_names)} 个类别的前景。")

    def _generate_single_scene(self):
        """生成单个合成场景及其标注 (逻辑不变)"""
        background = next(self.bg_cycler).copy()
        protagonist_class = next(self.class_cycler)
        placed_boxes, yolo_labels = [], []
        num_units = random.randint(self.config["MIN_UNITS_PER_IMAGE"], self.config["MAX_UNITS_PER_IMAGE"])
        unit_classes_to_place = [protagonist_class] + random.choices(self.class_names, k=num_units - 1)
        random.shuffle(unit_classes_to_place)

        for class_name in unit_classes_to_place:
            if not self.assets.get(class_name): continue
            asset = random.choice(self.assets[class_name])
            asset_h, asset_w = asset.shape[:2]

            for _ in range(self.config["MAX_PLACEMENT_ATTEMPTS"]):
                x = random.randint(0, self.bg_img_w - asset_w)
                y = random.randint(0, self.bg_img_h - asset_h)
                new_box = [x, y, x + asset_w, y + asset_h]
                is_colliding = any(calculate_iou(new_box, placed_box) > self.config["OVERLAP_THRESHOLD_IOU"] for placed_box in placed_boxes)
                
                if not is_colliding:
                    background = paste_transparent(background, asset, (x, y))
                    placed_boxes.append(new_box)
                    class_id = self.class_to_id[class_name]
                    cx, cy = (x + asset_w / 2) / self.bg_img_w, (y + asset_h / 2) / self.bg_img_h
                    w, h = asset_w / self.bg_img_w, asset_h / self.bg_img_h
                    yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    break
        return background, yolo_labels

    def run(self):
        """执行完整的数据合成流水线"""
        self._prepare_environment()
        self._load_assets()
        
        self.bg_cycler = cycle(self.backgrounds)
        self.class_cycler = cycle(self.class_names)
        
        print("--- [3/3] 开始生成合成数据... ---")
        num_total = self.config["NUM_IMAGES_TO_GENERATE"]

        for i in tqdm(range(num_total), desc="合成进度"):
            scene, labels = self._generate_single_scene()
            filename_stem = f"synth_unit_{i:06d}"
            
            # 【核心修改】直接将 .png 和 .txt 文件保存到同一个输出目录
            cv2.imwrite(str(self.output_dir / f"{filename_stem}.png"), scene)
            if labels:
                with open(self.output_dir / f"{filename_stem}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(labels))
                    
        print("\n" + "="*50)
        print("🎉🎉🎉 数据合成任务成功完成！ 🎉🎉🎉")
        print(f"   - '原始'数据集已直接生成于: {self.output_dir.resolve()}")
        print("   - 现在您可以直接使用此路径进行训练。")
        print("="*50)

if __name__ == '__main__':
    synthesizer = CombatUnitSynthesizer(CONFIG)
    synthesizer.run()