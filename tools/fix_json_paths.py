# 文件: fix_json_paths.py
# 职责: [数据修复工具] 扫描指定目录下的所有 LabelMe JSON 文件，
#       并将其内部的 'imagePath' 字段修正为与 JSON 文件本身同名。
#       例如：确保 0001.json 内部的 imagePath 是 "0001.png"。

import json
from pathlib import Path
import sys

def repair_labelme_paths(directory: Path):
    """
    遍历目录，修复所有 LabelMe JSON 文件中的 imagePath 字段。
    """
    print("--- 🚀 启动 LabelMe 数据完整性修复程序 ---")
    if not directory.is_dir():
        print(f"❌ 错误: 找不到目标目录: {directory}")
        return

    json_files = sorted(list(directory.glob("*.json")))
    if not json_files:
        print(f"⚠️ 在 {directory} 中未找到任何 .json 文件。")
        return

    print(f"🔍 正在扫描 {len(json_files)} 个 .json 文件...")
    
    files_repaired = 0
    files_skipped = 0

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 构造正确的 imagePath 名称 (与 json 文件同名，后缀为 .png)
            correct_image_path = f"{json_path.stem}.png"

            # 检查 'imagePath' 字段是否存在且需要修复
            if 'imagePath' in data and data['imagePath'] != correct_image_path:
                old_path = data['imagePath']
                print(f"🔧 正在修复: {json_path.name}")
                print(f"   - 旧路径: '{old_path}'")
                print(f"   - 新路径: '{correct_image_path}'")
                
                # 执行修正
                data['imagePath'] = correct_image_path
                
                # 写回文件，使用 indent 美化格式
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                files_repaired += 1
            else:
                files_skipped += 1

        except json.JSONDecodeError:
            print(f"⚠️ 警告: 文件 '{json_path.name}' 不是有效的JSON格式，已跳过。")
        except Exception as e:
            print(f"❌ 处理文件 '{json_path.name}' 时发生未知错误: {e}")

    print("\n" + "="*50)
    print("--- ✅ 修复完成 ---")
    print(f"   - 共扫描文件: {len(json_files)}")
    print(f"   - 成功修复文件: {files_repaired}")
    print(f"   - 无需修复/跳过: {files_skipped}")
    print("="*50)


if __name__ == '__main__':
    # --- 【请在这里配置您的数据目录】 ---
    # 将此路径修改为您包含 .json 和 .png 文件的实际目录
    target_directory_path = Path("/home/zhz/deepl/data/raw/yolo_ui_elements_v1")
    
    # 运行修复程序
    repair_labelme_paths(target_directory_path)