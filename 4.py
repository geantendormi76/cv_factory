
import os
import sys
from pathlib import Path

# --- 1. 路径定义 ---
BASE_DIR = Path.cwd()
README_PATH = BASE_DIR / "cv-README.md"

# --- 2. 重构后的 README 内容 ---
readme_content = r'''# 🚀 MHXY AI 模型工厂 (Computer Vision Factory)

> **核心理念：实验即配置 (Experiment as Code)**
> 你不需要懂代码，只需要修改一份配置文件 (`.yaml`)，即可驱动从数据处理、模型训练到最终导出的全自动化流水线。

---

## 🚦 第一步：选择你的数据路径 (Workflow Selector)

请根据你手头拥有的数据情况，选择 **路径 A** 或 **路径 B**。

### 🅰️ 路径 A：我有游戏截图，想自动生成数据 (推荐)
*适用场景：识别背包、仓库、摊位等网格状、图标固定的界面。效率最高。*

1.  **素材提取**
    *   将原始游戏截图放入 `data/raw/`。
    *   运行提取工具：`python tools/batch_icon_extractor.py`
    *   *操作：左键点选图标，Ctrl+左键排除背景，S键保存。*
2.  **网格校准**
    *   准备一张空背包截图 `data/assets/backgrounds/empty.png`。
    *   运行校准工具：`python tools/grid_calibrator.py`
    *   *操作：确认红框完美覆盖格子，如有偏差修改脚本内的坐标参数。*
3.  **一键合成**
    *   运行合成工具：`python tools/inventory_synthesizer.py`
    *   *结果：系统将自动生成 2000+ 张带完美标注的训练数据至 `data/raw/synthetic_inventory_v1`。*

### 🅱️ 路径 B：我有 LabelMe/YOLO 标注数据 (传统)
*适用场景：识别动态特效、不规则物体、或非网格界面。*

1.  **数据准备**
    *   将图片 (`.jpg/.png`) 和 标签 (`.json/.txt`) 放入 `data/raw/manual_dataset_v1` (文件夹名自定义)。
2.  **格式转换 (如果是 LabelMe)**
    *   如果你的标注是 `.json` 格式，系统会在训练时自动转换，无需额外操作。

---

## ⚙️ 第二步：填写“实验指令单” (Configuration)

打开项目根目录下的 **`yolo.yaml`**，这是你唯一需要修改的文件。

```yaml
# 1. 给本次训练起个名字 (必须唯一)
project_name: "yolo_daoju"
run_name: "v1_synthetic_test"

# 2. 告诉系统数据在哪里 (最重要！)
# -> 如果是路径 A (合成)，填合成后的目录:
source_data_dir: "/home/zhz/cv_factory/data/raw/synthetic_inventory_v1"
# -> 如果是路径 B (人工)，填你上传的目录:
# source_data_dir: "/home/zhz/cv_factory/data/raw/manual_dataset_v1"

# 3. 定义你要识别什么 (ID: 名字)
class_names:
  0: '桃花'
  1: '飞行符'
  # ...

# 4. 训练参数 (小白仅需关注以下几项)
hyperparameters:
  epochs: 150       # 训练轮数，建议 100-300
  batch: 32         # 显存够大可以改 64
  
  # --- 针对游戏UI的特殊设置 ---
  # 如果是固定图标(路径A)，必须设为 0.0 (禁止旋转/翻转)
  # 如果是怪物/特效(路径B)，可以设为 0.5 (允许增强)
  degrees: 0.0      
  fliplr: 0.0       
  scale: 0.05       # 图标大小固定，缩放要小
```

---

## ▶️ 第三步：一键启动 (Start)

配置完成后，在终端运行以下命令。系统会自动清洗数据、划分验证集、开始训练、并导出模型。

```bash
python pipeline.py --config yolo.yaml
```

*   **训练产物位置：** `runs/<project_name>/<run_name>/` (包含图表、日志)
*   **最终模型位置：** `saved/models/<run_name>.onnx` (可直接部署)

---

## 🎯 第四步：验货 (Validation)

**永远不要只看训练数据的分数，要看真实截图的效果。**

1.  确保 `data/raw/` 下有一些真实的游戏截图。
2.  运行可视化验证工具：
    ```bash
    python tools/validate_real.py
    ```
3.  **查看结果：** 打开 `AAA/real_world_validation_high_conf/` 目录。
    *   **绿色框**：识别出的物体。
    *   **HUD显示**：标签位于框内，半透明背景，不遮挡画面。
    *   *注：默认只显示置信度 > 0.90 的结果。*

---

## 📂 附录：项目结构速查

*   `configs/`: 存放 `main_config.yaml` (系统级配置，一般不动)。
*   `data/`:
    *   `assets/`: 合成用的素材 (图标、背景)。
    *   `raw/`: **你的工作区** (放截图、放标注数据)。
    *   `processed/`: 系统生成的临时数据 (可随时删除)。
*   `tools/`: **工具箱**
    *   `batch_icon_extractor.py`: 抠图工具。
    *   `inventory_synthesizer.py`: 合成工具。
    *   `validate_real.py`: 验证工具。
*   `yolo.yaml`: **实验指令单 (核心)**。
*   `pipeline.py`: **启动脚本**。
'''

# --- 3. 执行写入 ---
def update_readme():
    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"✅ 文档已重构: {README_PATH}")
    print("   - 结构优化：路径A (合成) vs 路径B (人工) 分流")
    print("   - 语言风格：面向小白，指令清晰")
    print("   - 核心保留：保留了所有工具的调用说明")

if __name__ == "__main__":
    update_readme()
