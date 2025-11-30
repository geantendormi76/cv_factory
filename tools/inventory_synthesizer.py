
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# --- 路径适配 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ================= 配置区域 =================
CONFIG = {
    "icons_dir": PROJECT_ROOT / "data/assets/icons",
    "bg_dir": PROJECT_ROOT / "data/assets/backgrounds",
    "output_dir": PROJECT_ROOT / "data/raw/synthetic_inventory_v1",
    "total_images": 2000, 
    
    # 概率设置
    "prob_empty": 0.3,      # 30% 概率格子是空的
    "prob_distractor": 0.2, # 20% 概率放干扰物 (不产生标签)
    "prob_valid": 0.5,      # 50% 概率放正样本 (产生标签)

    # 网格参数 (保持你之前校准过的)
    "grid": {
        "start_x": 384,
        "start_y": 187,
        "cell_w": 53,
        "cell_h": 53,
        "cols": 5,
        "rows": 4,
        "gap_x": 2,
        "gap_y": 2
    }
}
# ===========================================

def overlay_image_alpha(img, img_overlay, x, y, w, h):
    # 随机缩放 (0.85 - 0.95)
    scale = random.uniform(0.85, 0.95)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 保持长宽比
    h_src, w_src = img_overlay.shape[:2]
    aspect = w_src / h_src
    if new_w / new_h > aspect:
        new_w = int(new_h * aspect)
    else:
        new_h = int(new_w / aspect)

    try:
        img_overlay = cv2.resize(img_overlay, (new_w, new_h))
    except:
        return img

    # 居中 + 随机微调
    x += (w - new_w) // 2 + random.randint(-2, 2)
    y += (h - new_h) // 2 + random.randint(-2, 2)

    y1, y2 = y, y + new_h
    x1, x2 = x, x + new_w
    
    # 边界检查
    if y1 < 0 or y2 > img.shape[0] or x1 < 0 or x2 > img.shape[1]: return img

    # Alpha 混合
    if img_overlay.shape[2] == 4:
        alpha_s = img_overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * img_overlay[:, :, c] +
                                    alpha_l * img[y1:y2, x1:x2, c])
    else:
        # 如果素材没有透明通道，直接覆盖 (不推荐，但防报错)
        img[y1:y2, x1:x2] = img_overlay
        
    return img

def main():
    icons_dir = Path(CONFIG["icons_dir"])
    bg_dir = Path(CONFIG["bg_dir"])
    output_dir = Path(CONFIG["output_dir"])
    
    # 1. 加载素材并分类
    valid_icons = []   # 正样本 (ID < 900)
    distractors = []   # 干扰物 (ID >= 900)
    
    print(f"--- 正在加载素材: {icons_dir} ---")
    for p in icons_dir.glob("*.png"):
        try:
            # 解析文件名: "999_name_xxx.png" -> ID=999
            cls_id = int(p.name.split('_')[0])
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            if cls_id >= 900:
                distractors.append(img)
            else:
                valid_icons.append({'img': img, 'id': cls_id})
        except: pass
            
    print(f"   ✅ 正样本数量: {len(valid_icons)}")
    print(f"   ✅ 干扰物数量: {len(distractors)}")
    
    backgrounds = [cv2.imread(str(p)) for p in bg_dir.glob("*.png")]
    if not valid_icons or not backgrounds:
        print("❌ 错误: 缺少正样本或背景图！")
        return

    # 2. 准备输出
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # 3. 增强流水线
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        A.Blur(blur_limit=3, p=0.1),
    ])

    # 4. 合成循环
    grid = CONFIG["grid"]
    print(f"🚀 开始合成 {CONFIG['total_images']} 张含干扰物的训练数据...")
    
    for i in tqdm(range(CONFIG['total_images'])):
        bg = random.choice(backgrounds).copy()
        bg = transform(image=bg)['image']
        h_bg, w_bg = bg.shape[:2]
        
        labels = []
        
        for r in range(grid["rows"]):
            for c in range(grid["cols"]):
                # 决策：放什么？
                rand = random.random()
                
                # 计算格子坐标
                x = grid["start_x"] + c * (grid["cell_w"] + grid["gap_x"])
                y = grid["start_y"] + r * (grid["cell_h"] + grid["gap_y"])
                
                if rand < CONFIG["prob_empty"]:
                    # 情况1: 空格子
                    continue
                    
                elif rand < (CONFIG["prob_empty"] + CONFIG["prob_distractor"]) and len(distractors) > 0:
                    # 情况2: 放干扰物 (只贴图，不生成标签)
                    icon_img = random.choice(distractors)
                    bg = overlay_image_alpha(bg, icon_img, x, y, grid["cell_w"], grid["cell_h"])
                    # 【关键】这里不 append labels
                    
                else:
                    # 情况3: 放正样本 (贴图 + 生成标签)
                    icon_data = random.choice(valid_icons)
                    bg = overlay_image_alpha(bg, icon_data['img'], x, y, grid["cell_w"], grid["cell_h"])
                    
                    # 生成 YOLO 标签
                    box_w = grid["cell_w"] * 0.9
                    box_h = grid["cell_h"] * 0.9
                    cx = (x + grid["cell_w"]/2) / w_bg
                    cy = (y + grid["cell_h"]/2) / h_bg
                    nw = box_w / w_bg
                    nh = box_h / h_bg
                    
                    labels.append(f"{icon_data['id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # 保存
        name = f"syn_{i:05d}"
        cv2.imwrite(str(output_dir / f"{name}.jpg"), bg)
        with open(output_dir / f"{name}.txt", 'w') as f:
            f.write("\n".join(labels))
            
    print(f"\n✅ 合成完成！数据已保存至: {output_dir}")

if __name__ == "__main__":
    main()
