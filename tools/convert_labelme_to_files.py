# 文件: tools/convert_labelme_to_files.py (新创建)
# 职责: [一次性工具] 扫描指定目录，将内嵌图像数据的旧版LabelMe JSON文件，
#       转换为“独立的.png图片文件 + 干净的.json标注文件”的新格式。

import json
import base64
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image
import io

def convert_directory(source_dir: Path):
    """
    转换整个目录中的LabelMe JSON文件。
    """
    print(f"--- 🚀 启动LabelMe格式转换器 ---")
    print(f"扫描目录: {source_dir}")

    json_files = list(source_dir.glob("*.json"))
    if not json_files:
        print("⚠️ 未找到任何.json文件，无需转换。")
        return

    for json_path in tqdm(json_files, desc="转换进度"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查是否是需要转换的旧格式
            if 'imageData' not in data or not data['imageData']:
                continue # 如果没有图像数据，说明是新格式或空文件，跳过

            # 1. 解码并保存图像
            image_data = base64.b64decode(data['imageData'])
            image = Image.open(io.BytesIO(image_data))
            
            # 确保图像是RGB格式
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_path_str = data.get('imagePath', f"{json_path.stem}.png")
            image_save_path = source_dir / image_path_str
            image.save(image_save_path, "PNG")

            # 2. 清理并重写JSON文件
            data['imageData'] = None # 清空图像数据
            data['imagePath'] = image_path_str # 确保路径是相对的
            
            # 以更易读的格式写回
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"\n❌ 处理文件 {json_path.name} 时出错: {e}")

    print(f"--- ✅ 转换完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将内嵌图像的LabelMe JSON转换为文件格式。")
    parser.add_argument("source_directory", type=str, help="包含.json文件的源目录路径。")
    args = parser.parse_args()
    
    source_path = Path(args.source_directory)
    if not source_path.is_dir():
        print(f"错误: 目录不存在 -> {source_path}")
    else:
        convert_directory(source_path)