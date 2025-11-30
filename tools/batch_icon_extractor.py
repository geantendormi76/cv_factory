
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse
import yaml

# --- 路径适配 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("❌ 错误: 未安装 segment-anything。")
    sys.exit(1)

# --- 配置 ---
SAM_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda"

# --- 全局变量 ---
input_points = []
input_labels = []

def print_usage():
    """打印操作指南"""
    print("\n" + "="*60)
    print("🎮  SAM 强力抠图工具 (无冲突版) - 操作指南")
    print("="*60)
    print("🖱️ 【左键点击】         添加 [正向点] (这是物体)")
    print("🖱️ 【Ctrl + 左键】      添加 [负向点] (这是背景)")
    print("🖱️ 【鼠标中键(滚轮)】   添加 [负向点] (备用方案)")
    print("-" * 60)
    print("⌨️ 【S 键】     保存当前抠图")
    print("⌨️ 【R 键】     重置当前选择")
    print("⌨️ 【A 键】     <-- 上一张")
    print("⌨️ 【D 键】     --> 下一张")
    print("⌨️ 【Q 键】     退出")
    print("="*60 + "\n")

def mouse_callback(event, x, y, flags, param):
    global input_points, input_labels
    
    # 1. 添加负向点 (背景)
    # 逻辑: 按下中键，或者 按下左键的同时按住了Ctrl
    if event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY)):
        input_points.append([x, y])
        input_labels.append(0) # Label 0 = 背景
        print(f"  ➖ 添加负向点 (背景): ({x}, {y})")
        
    # 2. 添加正向点 (主体)
    # 逻辑: 仅按下左键 (且没有按Ctrl)
    elif event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])
        input_labels.append(1) # Label 1 = 主体
        print(f"  ➕ 添加正向点 (主体): ({x}, {y})")

def main():
    print_usage()
    
    raw_dir = PROJECT_ROOT / "data" / "raw" 
    output_base = PROJECT_ROOT / "data" / "assets" / "icons"
    
    if not raw_dir.exists():
        print(f"❌ 错误: 图片目录不存在: {raw_dir}")
        return
    output_base.mkdir(parents=True, exist_ok=True)

    print("\n⏳ 正在加载 SAM 模型 (请稍候)...")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        predictor = SamPredictor(sam)
        print("✅ 模型加载完毕！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 加载类别名
    class_names = {}
    yaml_path = PROJECT_ROOT / "yolo.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                class_names = cfg.get('class_names', {})
        except: pass

    images = sorted(list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg")))
    if not images:
        print("❌ 目录下没有找到图片！")
        return
        
    current_idx = 0
    total_images = len(images)
    
    while True:
        if current_idx < 0: current_idx = 0
        if current_idx >= total_images: current_idx = total_images - 1
        
        img_path = images[current_idx]
        print(f"\n[{current_idx+1}/{total_images}] 正在处理: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None: 
            print("   ⚠️ 无法读取，跳过。")
            current_idx += 1
            continue
        
        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 更新窗口标题提示
        window_name = f"Extractor (Left:Add, Ctrl+Left:Remove)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        global input_points, input_labels
        input_points = []
        input_labels = []
        
        while True:
            vis = image.copy()
            mask = None
            
            # 绘制点
            for pt, label in zip(input_points, input_labels):
                # 正向点=绿色实心，负向点=红色实心
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(vis, tuple(pt), 5, color, -1)
                # 给负向点加个白圈，更明显
                if label == 0:
                    cv2.circle(vis, tuple(pt), 5, (255, 255, 255), 1)
            
            # 实时预测
            if len(input_points) > 0:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(input_points),
                    point_labels=np.array(input_labels),
                    multimask_output=False
                )
                mask = masks[0]
                # 绿色半透明高亮
                vis[mask] = vis[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'): # Save
                if mask is not None:
                    if class_names:
                        print("\n📋 可选类别:")
                        for cid, cname in class_names.items():
                            print(f"   [{cid}] {cname}")
                    
                    try:
                        cls_id_str = input("👉 请输入类别ID (数字): ")
                        cls_id = int(cls_id_str)
                        cls_name = class_names.get(cls_id, "unknown")
                        
                        y_idx, x_idx = np.where(mask)
                        if len(y_idx) == 0: continue

                        y_min, y_max = y_idx.min(), y_idx.max()
                        x_min, x_max = x_idx.min(), x_idx.max()
                        
                        b, g, r = cv2.split(image)
                        alpha = (mask * 255).astype(np.uint8)
                        rgba = cv2.merge([b, g, r, alpha])
                        
                        crop = rgba[y_min:y_max+1, x_min:x_max+1]
                        
                        # 文件名增加时间戳或随机数防止覆盖
                        import time
                        save_name = f"{cls_id}_{cls_name}_{img_path.stem}_{int(time.time())}.png"
                        save_path = output_base / save_name
                        cv2.imwrite(str(save_path), crop)
                        print(f"✅ 成功保存: {save_name}")
                        
                        input_points = []
                        input_labels = []
                        
                    except ValueError:
                        print("❌ 输入无效")
                else:
                    print("⚠️ 请先点击物体")

            elif key == ord('r'): # Reset
                input_points = []
                input_labels = []

            elif key == ord('d'): # Next
                current_idx += 1
                break
            
            elif key == ord('a'): # Prev
                current_idx -= 1
                break
                
            elif key == ord('q'): # Quit
                sys.exit(0)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
