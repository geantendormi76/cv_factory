# 文件: pipeline.py (V2.0 - 健壮性强化版)
# 职责: 作为模型工厂的中央协调器，提供清晰的错误报告和灵活的路径处理，
#       解析实验配置，并按顺序驱动整个端到端的训练流水线。

import argparse
import json
import yaml
import sys
import importlib
from pathlib import Path
from typing import Dict, Any

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    print("错误: 核心依赖 pydantic 未安装。请运行: pip install pydantic")
    sys.exit(1)

# [IMPROVEMENT] 配置模型现在只负责结构和类型，不负责路径存在性校验。
# 校验的职责转移到拥有更多上下文的 Orchestrator 中。
class PipelineConfig(BaseModel):
    project_name: str
    run_name: str
    task_type: str
    source_data_dir: str # 从 DirectoryPath 改为 str，以增加灵活性
    base_model: str
    class_names: Dict[int, str]
    hyperparameters: Dict[str, Any]
    onnx_output_name: str


class Orchestrator:
    def __init__(self, config_path: Path):
        print(f"--- [1/5] 初始化流水线: 加载配置文件 ---")
        if not config_path.is_file():
            raise FileNotFoundError(f"指定的配置文件不存在: {config_path}")
        
        self.project_root = Path(__file__).resolve().parent
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            self.config = PipelineConfig.model_validate(raw_config)
            print(f"✅ 配置文件 '{config_path.name}' 结构验证成功。")
        except ValidationError as e:
            print(f"❌ 配置文件格式错误，请检查以下问题:\n{e}")
            raise
        except Exception as e:
            print(f"❌ 配置文件读取失败: {e}")
            raise
        
        # [ROBUSTNESS] 新增独立的路径解析和验证步骤
        self._resolve_and_validate_paths()

        self.processed_data_dir = self.project_root / "data" / "processed" / f"{self.config.run_name}_dataset"
        self.run_dir = self.project_root / "runs" / self.config.project_name / self.config.run_name
        self.temp_yolo_config_path = None

    def _resolve_and_validate_paths(self):
        """[ROBUSTNESS] 智能解析并验证所有关键路径。"""
        print("   - 正在验证路径...")
        
        # 处理 source_data_dir
        source_path = Path(self.config.source_data_dir)
        if not source_path.is_absolute():
            # 如果是相对路径，则相对于项目根目录进行解析
            source_path = self.project_root / source_path
        
        if not source_path.exists() or not source_path.is_dir():
            raise FileNotFoundError(f"关键错误: 解析后的源数据目录不存在或不是一个目录!\n   - 解析路径: {source_path}")
        
        # 将验证后的绝对路径写回配置对象，供后续使用
        self.config.source_data_dir = source_path
        print(f"   - 源数据目录验证成功: {self.config.source_data_dir}")

    def _prepare_environment(self):
        """准备运行环境，创建必要的目录。"""
        print(f"\n--- [2/5] 准备运行环境 ---")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"   - 数据处理目录: {self.processed_data_dir}")
        print(f"   - 训练输出目录: {self.run_dir}")
        print("✅ 环境准备就绪。")

    def _generate_yolo_config(self) -> Path:
        """根据实验配置动态生成一个临时的YOLO训练配置文件。"""
        yolo_params = self.config.hyperparameters.copy()
        
        yolo_params['model'] = self.config.base_model
        yolo_params['project'] = str(self.project_root / "runs" / self.config.project_name)
        yolo_params['name'] = self.config.run_name
        yolo_params['data'] = str(self.processed_data_dir / "dataset.yaml")
        
        self.temp_yolo_config_path = self.run_dir / "_temp_yolo_config.yaml"
        with open(self.temp_yolo_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yolo_params, f, sort_keys=False, allow_unicode=True)
        
        print(f"   - 动态生成YOLO配置文件: {self.temp_yolo_config_path}")
        return self.temp_yolo_config_path

    def run_data_pipeline(self):
        """执行数据准备流水线。"""
        print(f"\n--- [3/5] 执行数据准备流水线 ---")
        with open(self.project_root / "task_registry.json", 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        task_info = registry['tasks'].get(self.config.task_type)
        if not task_info:
            raise ValueError(f"任务类型 '{self.config.task_type}' 未在 task_registry.json 中注册。")
        
        print(f"   - 任务类型: {self.config.task_type} -> {task_info['description']}")

        module = importlib.import_module(task_info['data_builder_module'])
        builder_func = getattr(module, task_info['data_builder_func'])
        
        builder_func(
            source_dir=self.config.source_data_dir,
            output_dir=self.processed_data_dir,
            class_names=self.config.class_names
        )
        print(f"✅ 数据准备流水线执行成功。")

    def run_training_pipeline(self):
        """执行模型训练和导出流水线。"""
        print(f"\n--- [4/5] 执行模型训练与导出流水线 ---")
        yolo_config_path = self._generate_yolo_config()
        
        from trainer.trainer import Trainer
        
        trainer = Trainer(yolo_config_path)
        trainer.train()
        trainer.export_model(onnx_target_name=self.config.onnx_output_name)
        print(f"✅ 模型训练与导出成功。")

    def _cleanup(self):
        """清理临时文件。"""
        print(f"\n--- [5/5] 清理临时文件 ---")
        if self.temp_yolo_config_path and self.temp_yolo_config_path.exists():
            self.temp_yolo_config_path.unlink()
            print(f"   - 已删除临时文件: {self.temp_yolo_config_path}")
        print("✅ 清理完成。")

    def run(self):
        """按顺序执行完整的端到端流水线。"""
        try:
            self._prepare_environment()
            self.run_data_pipeline()
            self.run_training_pipeline()
        finally:
            self._cleanup()
        
        print("\n" + "="*80)
        print("🎉🎉🎉 流水线执行成功完成！ 🎉🎉🎉")
        print(f"   - 训练产物位于: {self.run_dir}")
        print(f"   - 最终ONNX模型位于: {self.project_root / 'saved/models' / self.config.onnx_output_name}")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="MHXY AI Model Factory - Central Pipeline")
    parser.add_argument(
        '-c', '--config',
        type=Path,
        required=True,
        help='指向实验配置文件 (run_config.yaml) 的路径'
    )
    args = parser.parse_args()

    try:
        orchestrator = Orchestrator(config_path=args.config)
        orchestrator.run()
    except Exception as e:
        # [ROBUSTNESS] 关键改进：打印完整、详细的错误追溯信息！
        print(f"\n--- ❌ 流水线执行过程中发生致命错误 ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()