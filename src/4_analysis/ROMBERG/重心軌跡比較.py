import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
import sys
from pathlib import Path
src_root = Path(__file__).resolve().parents[2]
sys.path.append(str(src_root))
from utils.config import get_project_root


# =========================================================
# ユーザー設定：比較したい2名の情報を入力
# =========================================================
# 【注意】フォルダ名が "NOCIPN" の場合はここは変更しないでください（読み込みに失敗します）
# 表示だけを "Non-CIPN" に自動変換するようにコード側で修正しました。

# 【設定1】比較対象 A (例: Non-CIPN群の患者)
subject_A = {
    "group": "NOCIPN",  # フォルダ名に合わせて NOCIPN のままにする
    "id": "P001"
}

# 【設定2】比較対象 B (例: CIPN群の患者)
subject_B = {
    "group": "CIPN",
    "id": "P002"
}

# 【設定3】その他の共通設定
TASK = "ROMBERG"
TRIAL = "T1"  # 1回目の試行のみ
BASE_ROOT = get_project_root()
# ---------------------------------------------------------
# 内部設定（変更不要）
# ---------------------------------------------------------
INPUT_ROOT = BASE_ROOT / "data" / "2_time_series_feature" / "main_research" / "CoG"
OUTPUT_ROOT = BASE_ROOT / "daily_results" / "20260126"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

FRAME_START = 200
# ファイル検索用パターン
pattern = re.compile(
    r"detected_(STUDENT|CIPN|NOCIPN)-(ROMBERG|ONELEG)-(P\d{3})-\d{8}-(EO|EC|L|R)-(T[123])-(C1|C2)_trim_CoG\.csv",
    re.IGNORECASE
)

# =========================================================
# 関数定義
# =========================================================
def apply_sg_filter(df, win=60, poly=3):
    df = df.copy()
    x = df["CoG_X_mm"].interpolate().to_numpy()
    y = df["CoG_Y_mm"].interpolate().to_numpy()
    df["CoG_X_mm_sg"] = savgol_filter(x, window_length=win, polyorder=poly)
    df["CoG_Y_mm_sg"] = savgol_filter(y, window_length=win, polyorder=poly)
    return df

def compute_path_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def compute_convex_area(x, y):
    try:
        hull = ConvexHull(np.column_stack((x, y)))
        return hull.volume
    except:
        return np.nan

def get_data(group, sid, cond):
    """
    指定された条件のデータを読み込み、X, Y配列を返す関数
    """
    # フォルダパス
    task_dir = os.path.join(INPUT_ROOT, group, sid, TASK)
    c1_dir = os.path.join(task_dir, "C1")
    c2_dir = os.path.join(task_dir, "C2")

    if not (os.path.exists(c1_dir) and os.path.exists(c2_dir)):
        print(f"Missing folder for {group}-{sid}")
        return None, None

    # ファイル検索
    files_c1 = glob.glob(os.path.join(c1_dir, "*.csv"))
    files_c2 = glob.glob(os.path.join(c2_dir, "*.csv"))

    f1_path, f2_path = None, None

    # C1検索
    for f in files_c1:
        m = pattern.search(os.path.basename(f))
        if m:
            g_ref, t_ref, pid_ref, c_ref, tr_ref, cam_ref = m.groups()
            if pid_ref == sid and c_ref == cond and tr_ref == TRIAL:
                f1_path = f
                break
    
    # C2検索
    for f in files_c2:
        m = pattern.search(os.path.basename(f))
        if m:
            g_ref, t_ref, pid_ref, c_ref, tr_ref, cam_ref = m.groups()
            if pid_ref == sid and c_ref == cond and tr_ref == TRIAL:
                f2_path = f
                break

    if f1_path is None or f2_path is None:
        print(f"Data not found: {sid} {cond} {TRIAL}")
        return None, None

    # 読み込みとフィルタリング
    try:
        df1 = apply_sg_filter(pd.read_csv(f1_path), win=21, poly=3)
        df2 = apply_sg_filter(pd.read_csv(f2_path), win=21, poly=3)
        
        n = min(len(df1), len(df2))
        if n < FRAME_START:
            return None, None

        # C1 -> X, C2 -> Y (元のコード準拠)
        x = df1["CoG_X_mm_sg"].iloc[FRAME_START:n].to_numpy()
        y = df2["CoG_X_mm_sg"].iloc[FRAME_START:n].to_numpy()
        
        # 重心を(0,0)に合わせる（比較しやすくするため）
        x = x - np.mean(x)
        y = y - np.mean(y)

        return x, y
    except Exception as e:
        print(f"Error processing {sid}: {e}")
        return None, None

# =========================================================
# メイン処理：比較プロット作成
# =========================================================
def main():
    # データを格納する辞書
    data_store = {
        "SubjectA": {"meta": subject_A, "EO": None, "EC": None},
        "SubjectB": {"meta": subject_B, "EO": None, "EC": None}
    }

    print(f"Loading Data for Comparison: {subject_A['id']} vs {subject_B['id']}")

    # -----------------------------
    # 1. データ読み込み
    # -----------------------------
    all_x = []
    all_y = []

    for key in ["SubjectA", "SubjectB"]:
        meta = data_store[key]["meta"]
        for cond in ["EO", "EC"]:
            x, y = get_data(meta["group"], meta["id"], cond)
            if x is not None:
                data_store[key][cond] = (x, y)
                all_x.extend(x)
                all_y.extend(y)
            else:
                print(f"Skipping {key} - {cond} due to missing data.")

    if not all_x:
        print("No data loaded. Check paths and IDs.")
        return

    # -----------------------------
    # 2. 軸スケールの決定 (Unified Scale)
    # -----------------------------
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    range_max = max(x_max - x_min, y_max - y_min) / 2.0 * 1.2 # 20%余裕
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    
    xlims = (center_x - range_max, center_x + range_max)
    ylims = (center_y - range_max, center_y + range_max)

    # -----------------------------
    # 3. プロット作成 (2行2列)
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # ★ 変更点: 表示用ラベルの作成（NOCIPN -> Non-CIPN）
    def get_display_group(grp):
        return "Non-CIPN" if grp == "NOCIPN" else grp

    group_a_disp = get_display_group(subject_A['group'])
    group_b_disp = get_display_group(subject_B['group'])

    subj_A_label = f"{group_a_disp}: {subject_A['id']}"
    subj_B_label = f"{group_b_disp}: {subject_B['id']}"
    
    cols = [subj_A_label, subj_B_label]
    rows = ["Open Eyes", "Closed Eyes"]

    # 列タイトルの設定
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=35, fontweight='bold', pad=20)

    # 行タイトルの設定
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, fontsize=35, fontweight='bold', labelpad=10)

    # ループで描画
    plot_map = [
        (0, 0, "SubjectA", "EO"), (0, 1, "SubjectB", "EO"),
        (1, 0, "SubjectA", "EC"), (1, 1, "SubjectB", "EC")
    ]

    for r, c, subj_key, cond in plot_map:
        ax = axes[r, c]
        data = data_store[subj_key][cond]

        # 軸設定
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.6)

        if data is None:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        x, y = data
        
        # 軌跡
        ax.plot(x, y, lw=2.5, color="royalblue", alpha=1)

        # Convex Hull
        try:
            hull = ConvexHull(np.column_stack((x, y)))
            pts = np.column_stack((x, y))[hull.vertices]
            ax.fill(pts[:, 0], pts[:, 1], color="salmon", alpha=0.2)
            ax.plot(pts[:, 0], pts[:, 1], color="tomato", lw=1.5)
            
            # 数値計算
            area = hull.volume
            length = compute_path_length(x, y)
            
            # 数値表示
            stats_text = f"Path: {length:.1f} mm\nArea: {area:.1f} mm²"
            ax.text(0.95, 0.95, stats_text, 
                    transform=ax.transAxes, 
                    ha='right', va='top', 
                    fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        except:
            pass

        # 中心
        ax.plot(0, 0, 'g+', markersize=10, markeredgewidth=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # 保存
    save_filename = f"Compare_{subject_A['id']}_vs_{subject_B['id']}_{TRIAL}.png"
    save_path = os.path.join(OUTPUT_ROOT, save_filename)
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved comparison plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()