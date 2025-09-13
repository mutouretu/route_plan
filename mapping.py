# durian_map_generator.py
# 生成随机榴莲地图（矩形地块），基地在地块正下方；保存 CSV 并绘制示意图。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

# --------------------
# 可调参数
# --------------------
SEED = 42                 # 随机种子（改它可重采样）
FIELD_RECT = (0.0, 0.0, 300.0, 240.0)  # (xmin, ymin, xmax, ymax)，单位：米
N_TREES = 180             # 榴莲树数量
MIN_SPACING = 5.0         # 树间最小间距（软约束，尽量满足）
BASE_OFFSET = 30.0        # 基地在地块下方的垂直距离（米）
CSV_PATH = "data/durian_map.csv"
PNG_PATH = "durian_map.png"

# --------------------
# 工具函数
# --------------------
def rejection_sample_points(xmin: float, ymin: float, xmax: float, ymax: float,
                            n: int, min_spacing: float, seed: int = 0) -> List[Tuple[float, float]]:
    """
    采用简单的拒绝采样生成点，尽量满足最小间距（软约束）。
    若在限定尝试次数内未放满，则直接补齐（不再检查间距）。
    """
    rng = np.random.default_rng(seed)
    points: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = 20000

    while len(points) < n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        ok = True
        # 仅和最近插入的部分点比较以提速（工程上足够）
        for (px, py) in points[-200:]:
            if (x - px) ** 2 + (y - py) ** 2 < (min_spacing ** 2):
                ok = False
                break
        if ok:
            points.append((x, y))

    # 若因间距限制未放满，直接补齐（允许少量近点）
    while len(points) < n:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        points.append((x, y))

    return points

# --------------------
# 主流程
# --------------------
def main():
    np.random.seed(SEED)

    xmin, ymin, xmax, ymax = FIELD_RECT
    # 基地放在底边中点下方 BASE_OFFSET 米
    base_x = (xmin + xmax) / 2.0
    base_y = ymin - BASE_OFFSET

    # 生成树坐标
    trees = rejection_sample_points(xmin, ymin, xmax, ymax,
                                    n=N_TREES, min_spacing=MIN_SPACING, seed=SEED)

    # 构建表
    df_trees = pd.DataFrame(trees, columns=["x", "y"])
    df_trees.insert(0, "type", "tree")
    df_trees.insert(1, "id", range(1, len(df_trees) + 1))

    df_base = pd.DataFrame([["base", 0, base_x, base_y]], columns=["type", "id", "x", "y"])
    df_all = pd.concat([df_base, df_trees], ignore_index=True)

    # 保存 CSV
    df_all.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV to: {CSV_PATH}")

    # 画图
    plt.figure(figsize=(8, 6))
    # 地块边界
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], color="#d18f00")
    # 树
    plt.scatter(df_trees["x"], df_trees["y"], s=15, label="durian tree", color="#d18f00")
    # 基地
    plt.scatter([base_x], [base_y], marker="^", s=60, label="UAV base", color="#3b82f6")

    plt.title("Random Durian Orchard Map (meters)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.grid(alpha=0.25)

    plt.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {PNG_PATH}")

if __name__ == "__main__":
    main()
