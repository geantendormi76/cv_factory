
import cv2
import numpy as np
from pathlib import Path
import sys

# --- 路径适配 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# 🔧 自动计算出的参数 (基于用户测量)
# ==========================================
GRID_CONFIG = {
    "start_x": 384,
    "start_y": 187,
    "cell_w": 53,
    "cell_h": 53,
    "cols": 5,
    "rows": 4,
    "gap_x": 2,
    "gap_y": 2
}
# ==========================================

def main():
    print("--- 📏 网格校准工具 (Auto-Calculated) ---")
    
    bg_dir = PROJECT_ROOT / "data" / "assets" / "backgrounds"
    bg_files = list(bg_dir.glob("*.png")) + list(bg_dir.glob("*.jpg"))
    
    if not bg_files:
        print(f"❌ 错误: 请先放一张背景图到 {bg_dir}")
        return

    bg_path = bg_files[0]
    print(f"正在使用背景图: {bg_path.name}")
    
    image = cv2.imread(str(bg_path))
    if image is None: return

    cfg = GRID_CONFIG
    vis = image.copy()
    
    print(f"当前参数: Start({cfg['start_x']},{cfg['start_y']}) | Size({cfg['cell_w']}x{cfg['cell_h']}) | Gap({cfg['gap_x']})")

    for r in range(cfg["rows"]):
        for c in range(cfg["cols"]):
            x = cfg["start_x"] + c * (cfg["cell_w"] + cfg["gap_x"])
            y = cfg["start_y"] + r * (cfg["cell_h"] + cfg["gap_y"])
            
            # 画矩形框 (红色, 宽度2)
            cv2.rectangle(vis, (x, y), (x + cfg["cell_w"], y + cfg["cell_h"]), (0, 0, 255), 2)
            # 画中心点
            cv2.circle(vis, (x + cfg["cell_w"]//2, y + cfg["cell_h"]//2), 2, (0, 255, 0), -1)

    # 缩放显示
    scale = 1.0
    if vis.shape[0] > 900: scale = 0.8
    vis_show = cv2.resize(vis, None, fx=scale, fy=scale)
    
    cv2.imshow("Grid Calibration", vis_show)
    print("\n👀 请检查红框是否完美覆盖了背包格子。")
    print("   按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
