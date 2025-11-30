
import cv2
import numpy as np
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ================= 配置区域 =================
MODEL_PATH = Path("runs/yolo_daoju_v1/yolo_daoju_v1/weights/best.pt")
SOURCE_DIR = Path("data/raw")
OUTPUT_DIR = Path("AAA/real_world_validation_high_conf") # 改个名，区分一下
FONT_PATH = Path("utils/fonts/SimHei.ttf")

# 【核心调整】置信度阈值
CONF_THRESHOLD = 0.90   # 只显示 90% 以上把握的目标
IOU_THRESHOLD = 0.6     # NMS 阈值

# HUD 显示设置
FONT_SIZE = 12
TEXT_PADDING = 2
BOX_THICKNESS = 2
MASK_ALPHA = 160
# ===========================================

def draw_detections_hud(image, results, font_path, font_size):
    """
    HUD风格绘制：半透明背景 + 框内显示
    """
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new('RGBA', image_pil.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_text = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    for box in results.boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label_name = results.names[cls_id]
        
        display_text = f"{label_name} {conf:.2f}"
        color_rgb = (0, 255, 0) # 绿色
        
        # 画框
        draw_overlay.rectangle([x1, y1, x2, y2], outline=color_rgb + (255,), width=BOX_THICKNESS)
        
        # 计算文字背景
        text_bbox = draw_text.textbbox((0, 0), display_text, font=font)
        text_w = text_bbox[2] - text_bbox[0] + TEXT_PADDING * 2
        text_h = text_bbox[3] - text_bbox[1] + TEXT_PADDING * 2
        
        text_x = x1 + BOX_THICKNESS
        text_y = y1 + BOX_THICKNESS
        
        # 画半透明背景
        bg_rect = [text_x, text_y, text_x + text_w, text_y + text_h]
        draw_overlay.rectangle(bg_rect, fill=(0, 0, 0, MASK_ALPHA))
        
        # 合并图层
        image_pil = Image.alpha_composite(image_pil, overlay)
        
    # 统一画文字
    draw_final = ImageDraw.Draw(image_pil)
    for box in results.boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, _, _ = coords
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label_name = results.names[cls_id]
        display_text = f"{label_name} {conf:.2f}"
        
        text_x = x1 + BOX_THICKNESS + TEXT_PADDING
        text_y = y1 + BOX_THICKNESS + TEXT_PADDING
        
        draw_final.text((text_x, text_y), display_text, fill=(255, 255, 255, 255), font=font)

    return cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

def main():
    print(f"--- 🚀 启动高置信度验证 (Conf >= {CONF_THRESHOLD}) ---")
    print(f"   - 模型: {MODEL_PATH}")
    print(f"   - 输出: {OUTPUT_DIR}")
    
    if not MODEL_PATH.exists():
        print(f"❌ 错误: 模型未找到")
        return

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    images = sorted(list(SOURCE_DIR.glob("*.png")) + list(SOURCE_DIR.glob("*.jpg")))

    print("\n--- 正在推理 ---")
    for i, img_path in enumerate(images):
        img0 = cv2.imread(str(img_path))
        if img0 is None: continue

        # 推理 (应用新的阈值)
        results = model.predict(img0, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
        
        # 绘制
        final_img = draw_detections_hud(img0, results, FONT_PATH, FONT_SIZE)
        
        # 保存
        save_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(save_path), final_img)
        
        print(f"[{i+1}/{len(images)}] {img_path.name}: {len(results.boxes)} 个目标 (Conf>={CONF_THRESHOLD})")

    print(f"\n✅ 过滤完成！请查看: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
