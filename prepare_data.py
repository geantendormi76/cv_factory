# 文件: prepare_data.py (V2.1 - 最终健壮版)
# 职责: 作为数据准备的统一入口，负责解析参数、注入上下文并调用加载器。

import argparse
import json
from pathlib import Path
import importlib
import copy

def main(config_path: Path, task_name: str):
    """
    主执行函数。
    Args:
        config_path (Path): 指向主配置文件的路径。
        task_name (str): 要执行的任务名称。
    """
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"❌ 错误: 解析配置文件 '{config_path}' 失败: {e}")
        return

    if task_name not in config['tasks']:
        print(f"❌ 错误: 在 '{config_path}' 中找不到任务 '{task_name}' 的定义。")
        print(f"   可用任务: {list(config['tasks'].keys())}")
        return

    task_config = config['tasks'][task_name]
    print(f"\n--- 启动数据准备任务: {task_config['description']} ---")
    
    data_builder_module_name = task_config['data_builder_module']
    data_builder_func_name = task_config['data_builder_func']
    
    try:
        # 上下文注入
        contextual_config = copy.deepcopy(config)
        contextual_config['current_task'] = task_name

        module = importlib.import_module(data_builder_module_name)
        prepare_data_func = getattr(module, data_builder_func_name)
        
        prepare_data_func(contextual_config)
        
        print(f"\n✅ 数据准备任务 '{task_name}' 完成！")
    except Exception as e:
        print(f"❌ 错误: 执行数据准备脚本时失败。")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # --- 【核心重构】优雅且健壮的命令行解析 ---
    
    # 1. 创建一个基础解析器，只用于获取配置文件路径
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('-c', '--config', default='config.json', type=str,
                             help='主配置文件路径 (默认: config.json)')
    
    # 2. 解析一次，只为了拿到 config 文件路径
    #    parse_known_args 会忽略它不认识的参数（比如 --task）
    args, _ = base_parser.parse_known_args()
    config_path = Path(args.config)

    # 3. 根据配置文件，动态确定合法的任务选项
    valid_tasks = []
    if config_path.is_file():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            valid_tasks = list(config_data.get('tasks', {}).keys())
        except Exception:
            # 如果配置文件解析失败，让后续的流程去处理错误
            pass

    # 4. 创建一个继承自基础解析器的、完整的解析器
    #    现在我们可以使用动态生成的 valid_tasks 来定义 choices
    parser = argparse.ArgumentParser(
        description='MHXY AI Model Factory - Data Preparation',
        parents=[base_parser] # 继承 -c/--config 参数
    )
    parser.add_argument(
        '-t', '--task', 
        type=str, 
        required=True, 
        choices=valid_tasks if valid_tasks else None, # 如果没加载到配置，则不限制choices
        help='要运行的任务名称 (从配置文件中读取)'
    )
    
    # 5. 进行最终的、完整的参数解析
    final_args = parser.parse_args()

    # 6. 调用主函数
    if not config_path.is_file():
        print(f"❌ 错误: 找不到配置文件 '{config_path}'")
    else:
        main(config_path, final_args.task)