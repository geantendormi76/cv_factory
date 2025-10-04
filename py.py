# 文件: create_digest.py (V3.0 - 双重过滤最终版)
# 职责: 一个专业的、可配置的、100%可靠的本地代码提取器。
#       通过“文件名黑名单”和“扩展名白名单”双重过滤，确保输出绝对纯净。

import os
from pathlib import Path

# ==================================================================
# 1. 核心配置：定义双重过滤规则
# ==================================================================

# 黑名单：无论扩展名是什么，都强制排除这些特定的文件名
EXCLUDE_FILENAMES = {
    "ag_environment.yml",
    "ocr_environment.yml",
    "yolo_environment.yml",
    # 未来可以添加其他已知的非文本文件
}

# 白名单：只包含这些后缀名的文件
TARGET_EXTENSIONS = {".py", ".yml", ".yaml", ".json"}

# 黑名单：彻底排除这些目录及其所有内容
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".idea", ".vscode", ".pytest_cache",
    "venv", ".venv", "env", "node_modules", "dist", "build",
    "site-packages", "logs", "data", "models", "tests"
}

# ==================================================================

def generate_code_digest(root_dir: Path, output_filename: str = "full_code_filtered.txt"):
    """
    遍历指定目录，通过双重过滤生成一个纯净的文本摘要。
    """
    print(f"🚀 开始分析项目: {root_dir}")
    
    target_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        
        for filename in filenames:
            # --- [核心修正：双重过滤逻辑] ---
            # 1. 优先检查文件名是否在黑名单中
            if filename in EXCLUDE_FILENAMES:
                continue # 如果是，立即跳过

            file_path = Path(dirpath) / filename
            
            # 2. 再检查文件扩展名是否在白名单中
            if file_path.suffix in TARGET_EXTENSIONS:
                target_files.append(file_path)

    target_files.sort()
    print(f"🔍 找到 {len(target_files)} 个核心代码/配置文件。")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("=== 项目核心文件结构 ===\n\n")
        f.write(f"{root_dir.name}\n")
        
        tree_structure = {}
        for path in target_files:
            parts = path.relative_to(root_dir).parts
            current_level = tree_structure
            for part in parts:
                if part not in current_level: current_level[part] = {}
                current_level = current_level[part]
        
        _write_tree(f, tree_structure)
        
        f.write("\n\n=== 核心文件内容 ===\n")
        for file_path in target_files:
            relative_path = file_path.relative_to(root_dir)
            formatted_path = str(relative_path).replace('\\', '/')
            
            f.write(f"\n================================================\n")
            f.write(f"FILE: {formatted_path}\n")
            f.write(f"================================================\n\n")
            try:
                content = file_path.read_text(encoding="utf-8", errors='ignore')
                f.write(content)
            except Exception as e:
                f.write(f"*** 无法读取文件: {e} ***\n")

    print(f"✅ 专业代码提取完成！")
    print(f"摘要已保存到: {output_filename}")


def _write_tree(file_handle, tree_structure, prefix=""):
    entries = sorted(tree_structure.keys())
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        file_handle.write(f"{prefix}{connector}{entry}\n")
        if tree_structure[entry]:
            new_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
            _write_tree(file_handle, tree_structure[entry], new_prefix)


if __name__ == "__main__":
    current_directory = Path.cwd()
    generate_code_digest(current_directory)