# 文件: train.py (V3.1 - 最终健壮版)
# 职责: 作为模型训练的统一入口，健壮地处理命令行参数。

import argparse
import json
from pathlib import Path
from trainer.trainer import Trainer

CONFIG_DIR = Path("configs")

def main(config_path: Path, task_name: str):
    """
    主执行函数。
    """
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ 错误: 解析配置文件 '{config_path}' 失败: {e}")
        return
        
    if task_name not in config['tasks']:
        print(f"❌ 错误: 在 '{config_path}' 中找不到任务 '{task_name}' 的定义。")
        return

    task_config = config['tasks'][task_name]
    print(f"\n--- 启动训练任务: {task_config['description']} ---")
    
    yolo_config_path = CONFIG_DIR / task_config['yolo_config_path']

    try:
        trainer_instance = Trainer(yolo_config_path)
        trainer_instance.train()
    except Exception as e:
        print(f"❌ 错误: 训练过程中发生失败。")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # --- 【核心重构】与 prepare_data.py 完全对齐的解析逻辑 ---
    
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('-c', '--config', default='config.json', type=str,
                             help='主配置文件路径 (默认: config.json)')
    
    args, _ = base_parser.parse_known_args()
    config_path = Path(args.config)

    valid_tasks = []
    if config_path.is_file():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            valid_tasks = list(config_data.get('tasks', {}).keys())
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description='MHXY AI Model Factory - Training',
        parents=[base_parser]
    )
    parser.add_argument(
        '-t', '--task', 
        type=str, 
        required=True, 
        choices=valid_tasks if valid_tasks else None,
        help='要运行的任务名称 (从配置文件中读取)'
    )
    
    final_args = parser.parse_args()

    if not config_path.is_file():
        print(f"❌ 错误: 找不到配置文件 '{config_path}'")
    else:
        main(config_path, final_args.task)