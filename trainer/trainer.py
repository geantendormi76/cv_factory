# 文件: trainer/trainer.py (V2.0 - 参数化解耦版)
# 职责: 封装与 Ultralytics YOLO 库的交互，负责执行训练和模型导出。
#       它完全由一个指定的 YOLO 配置文件驱动。

import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

class Trainer:
    """
    一个健壮的、配置驱动的YOLO模型训练器和导出器。
    """
    def __init__(self, yolo_config_path: Path):
        """
        初始化训练器。
        
        Args:
            yolo_config_path (Path): 指向为本次运行动态生成的YOLO .yaml配置文件的路径。
        """
        self.yolo_config_path = yolo_config_path
        if not self.yolo_config_path.is_file():
            raise FileNotFoundError(f"YOLO配置文件未找到: {self.yolo_config_path}")
            
        with open(self.yolo_config_path, 'r', encoding='utf-8') as f:
            self.yolo_config = yaml.safe_load(f)
        
        # 使用配置中定义的 'model' 键来初始化YOLO模型
        # 这可能是 "yolov8n.pt" 或一个指向 best.pt 的路径
        self.model = YOLO(self.yolo_config.get('model', 'yolov8n.pt'))

    def train(self):
        """
        使用加载的配置启动YOLOv8训练。
        """
        print(f"--- 使用配置文件 '{self.yolo_config_path.name}' 开始训练 ---")
        
        # 动态移除 'model' 键，因为它只用于初始化，不应传递给 train 方法
        # 'names' 字段也不需要，因为它已经包含在 data.yaml 中
        train_params = self.yolo_config.copy()
        train_params.pop('model', None)
        train_params.pop('names', None) 
        
        self.model.train(**train_params)
        print("--- ✅ 训练完成 ---")
        
    def export_model(self, onnx_target_name: str, model_to_export=None):
        """
        以高兼容性模式导出训练好的最佳模型为ONNX格式。

        Args:
            onnx_target_name (str): 最终导出的ONNX文件名 (例如: "my_model_v1.onnx")。
            model_to_export (Path, optional): 如果提供，则导出指定路径的模型，否则导出本次训练的最佳模型。
        """
        if model_to_export:
            model = YOLO(str(model_to_export))
            print(f"\n🚀 正在导出指定的模型: {model_to_export}")
        else:
            # 自动获取本次训练的最佳模型路径
            # self.model.trainer.best 在训练后会被赋值
            best_model_path = Path(self.model.trainer.best)
            model = self.model
            print(f"\n🏆 正在导出本次训练的最佳模型: {best_model_path}")

        print("🚀 正在以【最高兼容性】模式导出检测器模型...")
        export_params = {
            'format': 'onnx',
            'opset': 13,         
            'simplify': False,  # 禁用以避免环境兼容性问题
            'nms': False,       
            'dynamic': False,     
            'batch': 1,          
            'imgsz': self.yolo_config.get('imgsz', 640)
        }

        try:
            # 导出的模型会先生成在 runs/... 目录下
            onnx_path_temp = model.export(**export_params)
            
            # 将其移动到标准化的 `saved/models` 目录下，并重命名
            target_onnx_path = Path("saved/models") / onnx_target_name
            target_onnx_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(onnx_path_temp), str(target_onnx_path))
            
            print("\n" + "="*50)
            print("✅ 导出成功！")
            print(f"   已生成模型: {target_onnx_path}")
            print("="*50)
        except Exception as e:
            print(f"\n--- ❌ 导出失败！错误: {e} ---")
            import traceback
            traceback.print_exc()
            raise