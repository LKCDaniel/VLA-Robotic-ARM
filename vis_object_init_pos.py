import os
import matplotlib.pyplot as plt
from tqdm import trange
from util import read_json


def main():
    xs = []
    ys = []
    for episode_index in trange(500):
        data_path = os.path.join(os.path.dirname(__file__), "episodes", f"episode_{episode_index}", "robot_state.json")
        data = read_json(data_path)
        x = data["object_init_x"]
        y = data["object_init_y"]
        xs.append(x)
        ys.append(y)

    # ==================== 开始画图 ====================
    plt.figure(figsize=(10, 8))
    plt.scatter(xs, ys, c='dodgerblue', alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

    plt.xlabel("object_init_x", fontsize=14)
    plt.ylabel("object_init_y", fontsize=14)
    plt.title("300 Episodes - Object Initial Positions", fontsize=16, pad=20)

    # 让 x/y 轴比例相等，防止看起来被拉伸
    plt.axis('equal')

    # 可选：加网格、加边框
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 保存高清图（推荐）
    plt.savefig("object_init_positions.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()