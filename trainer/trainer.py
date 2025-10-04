# 文件: trainer/main_trainer.py (重构版)
import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil

class Trainer:
    def __init__(self, yolo_config_path: str):
        """
        初始化训练器。
        :param yolo_config_path: 指向特定任务的YOLO .yaml配置文件的路径。
        """
        self.yolo_config_path = Path(yolo_config_path)
        if not self.yolo_config_path.is_file():
            raise FileNotFoundError(f"YOLO配置文件未找到: {self.yolo_config_path}")
            
        with open(self.yolo_config_path, 'r') as f:
            self.yolo_config = yaml.safe_load(f)
        
        self.model = YOLO(self.yolo_config.get('model', 'yolo11n.pt'))

    def train(self):
        """
        使用加载的配置启动YOLOv8训练。
        """
        print(f"--- 使用配置文件 '{self.yolo_config_path.name}' 开始训练 ---")
        self.yolo_config.pop('names', None)
        self.model.train(**self.yolo_config)
        print("--- ✅ 训练完成 ---")
        
        # 训练完成后自动调用导出流程
        self.export_model()

# 文件: trainer/trainer.py

    def export_model(self, model_to_export=None):
        """
        以“黄金标准”原则导出指定的或训练好的最佳模型为ONNX格式。
        【最终修正】永久禁用 simplify=True 以避免环境兼容性导致的底层崩溃。
        """
        if model_to_export:
            model = YOLO(model_to_export)
            print(f"\n🚀 正在导出指定的模型: {model_to_export}")
        else:
            model = self.model
            best_model_path = Path(model.trainer.best)
            print(f"\n🏆 训练出的最佳模型: {best_model_path}")
        
        task_type = self.yolo_config.get('task', 'detect')
        
        if task_type == 'detect':
            print("🚀 正在以【最高兼容性】模式导出检测器模型...")
            target_name = "yolo_ui_elements_v1.onnx"
            export_params = {
                'format': 'onnx',
                'opset': 13,         
                'simplify': False,  # 【核心修正】禁用此项
                'nms': False,       
                'dynamic': False,     
                'batch': 1,          
                'imgsz': self.yolo_config.get('imgsz', 640)
            }
        # ... (classify 部分保持不变)
        elif task_type == 'classify':
            print("🚀 正在导出分类器模型...")
            target_name = "guaiwu_classifier.onnx"
            export_params = {
                'format': 'onnx',
                'opset': 12,
                'simplify': False, # 【核心修正】同样禁用
                'imgsz': self.yolo_config.get('imgsz', 64)
            }
        else:
            print(f"⚠️ 警告: 未知的任务类型 '{task_type}'，跳过模型导出。")
            return

        try:
            onnx_path = model.export(**export_params)
            target_onnx_path = Path("saved/models") / target_name
            target_onnx_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(onnx_path), str(target_onnx_path))
            
            print("\n" + "="*50)
            print("✅ 导出成功！")
            print(f"   已生成模型: {target_onnx_path}")
            print("="*50)
        except Exception as e:
            print(f"\n--- ❌ 导出失败！错误: {e} ---")
            import traceback
            traceback.print_exc()