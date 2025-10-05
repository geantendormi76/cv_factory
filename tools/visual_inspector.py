# 文件: visual_inspector.py
# 职责: [数据质量保证工具] 对一个已处理好的YOLO格式数据集进行可视化抽样检查，
#       将边界框和中文类别标签直接绘制在图片上，以供人工审查。

import cv2
import yaml
import random
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

class DatasetVisualInspector:
    """
    一个健壮的数据集可视化检查器。
    """
    def __init__(self, dataset_dir: Path, font_path: Path):
        """
        初始化检查器。

        Args:
            dataset_dir (Path): 指向已处理的数据集根目录 (包含 dataset.yaml 的目录)。
            font_path (Path): 指向用于显示中文标签的 .ttf 字体文件路径。
        """
        self.dataset_dir = dataset_dir
        self.font_path = font_path
        self.class_names = []
        self.colors = []

        # --- 核心初始化与验证 ---
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"❌ 错误: 数据集目录不存在 -> {self.dataset_dir}")
        
        if not self.font_path.is_file():
            raise FileNotFoundError(f"❌ 错误: 字体文件不存在 -> {self.font_path}")

        self._load_dataset_info()
        self._generate_class_colors()

    def _load_dataset_info(self):
        """加载 dataset.yaml 文件获取类别信息。"""
        yaml_path = self.dataset_dir / "dataset.yaml"
        if not yaml_path.is_file():
            raise FileNotFoundError(f"❌ 错误: 在数据集中未找到 'dataset.yaml' 文件。")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'names' not in data:
            raise KeyError("❌ 错误: 'dataset.yaml' 文件中缺少 'names' 字段。")
        
        # 支持字典 {0: 'name'} 和列表 ['name'] 两种格式
        if isinstance(data['names'], dict):
            self.class_names = [data['names'][i] for i in sorted(data['names'].keys())]
        elif isinstance(data['names'], list):
            self.class_names = data['names']
        else:
            raise TypeError("❌ 错误: 'names' 字段格式无法识别，应为字典或列表。")
            
        print(f"✅ 成功加载 {len(self.class_names)} 个类别: {self.class_names}")

    def _generate_class_colors(self):
        """为每个类别生成一个固定的、鲜艳的颜色。"""
        random.seed(42) # 固定随机种子，确保每次颜色一致
        for _ in self.class_names:
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            self.colors.append(color)

    def inspect(self, output_dir: Path, num_images: int = 10):
        """
        执行可视化检查。

        Args:
            output_dir (Path): 保存可视化结果的目录。
            num_images (int): 要随机抽样检查的图片数量。-1 表示检查所有图片。
        """
        print(f"\n--- 🚀 开始可视化检查 ---")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   - 结果将保存至: {output_dir}")

        image_files = list((self.dataset_dir / "images/train").glob("*.png")) + \
                      list((self.dataset_dir / "images/val").glob("*.png"))
        
        if not image_files:
            print("⚠️ 警告: 在 'images/train' 或 'images/val' 中未找到任何 .png 图片。")
            return

        if num_images != -1 and num_images < len(image_files):
            images_to_check = random.sample(image_files, num_images)
            print(f"   - 从 {len(image_files)} 张图片中随机抽取 {num_images} 张进行检查...")
        else:
            images_to_check = image_files
            print(f"   - 检查所有 {len(image_files)} 张图片...")

        for image_path in tqdm(images_to_check, desc="处理进度"):
            self._draw_boxes_on_image(image_path, output_dir)
        
        print("\n--- ✅ 检查完成 ---")

    def _draw_boxes_on_image(self, image_path: Path, output_dir: Path):
        """在单张图片上绘制其对应的所有边界框。"""
        label_path = self.dataset_dir / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ 警告: 无法读取图片 {image_path.name}，已跳过。")
            return
        
        img_h, img_w, _ = image.shape

        if not label_path.is_file():
            # 如果没有标签文件，也保存一份原图，便于对比
            cv2.imwrite(str(output_dir / f"{image_path.name}"), image)
            return

        # --- 使用 Pillow 绘制中文 ---
        # 1. OpenCV (BGR) -> Pillow (RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(str(self.font_path), size=15)

        with open(label_path, 'r') as f:
            for line in f:
                try:
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    class_id = int(class_id)

                    # 将 YOLO 格式坐标转换为左上角和右下角坐标
                    abs_cx = cx * img_w
                    abs_cy = cy * img_h
                    abs_w = w * img_w
                    abs_h = h * img_h
                    x1 = int(abs_cx - abs_w / 2)
                    y1 = int(abs_cy - abs_h / 2)
                    x2 = int(abs_cx + abs_w / 2)
                    y2 = int(abs_cy + abs_h / 2)

                    # 获取类别信息
                    label_text = self.class_names[class_id]
                    color = self.colors[class_id]

                    # 绘制边界框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # 绘制标签背景和文字
                    text_bbox = draw.textbbox((x1, y1 - 15), label_text, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((x1, y1 - 15), label_text, font=font, fill=(0, 0, 0))

                except (ValueError, IndexError) as e:
                    print(f"⚠️ 警告: 解析标签文件 {label_path.name} 中的行 '{line.strip()}' 时出错: {e}")

        # 2. Pillow (RGB) -> OpenCV (BGR)
        final_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"inspected_{image_path.name}"), final_image)


if __name__ == '__main__':
    # --- 【配置区】 ---
    # 1. 指向您要检查的数据集目录 (即 pipeline.py 的输出目录)
    DATASET_TO_INSPECT = Path("/home/zhz/deepl/data/processed/ui_elements_v1_run_dataset")
    
    # 2. 您的字体文件路径
    FONT_FILE = Path("/home/zhz/deepl/utils/fonts/SimHei.ttf")
    
    # 3. 可视化结果的输出目录
    INSPECTION_OUTPUT_DIR = Path("/home/zhz/deepl/inspection_results")
    
    # 4. 抽样检查的图片数量 (-1 表示检查全部)
    NUM_SAMPLES = 20
    # --- 【配置区结束】 ---

    try:
        inspector = DatasetVisualInspector(
            dataset_dir=DATASET_TO_INSPECT,
            font_path=FONT_FILE
        )
        inspector.inspect(
            output_dir=INSPECTION_OUTPUT_DIR,
            num_images=NUM_SAMPLES
        )
    except Exception as e:
        print(f"\n--- ❌ 质检工具运行时发生错误 ---")
        print(e)