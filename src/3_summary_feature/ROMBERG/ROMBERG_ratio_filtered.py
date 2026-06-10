import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
from pathlib import Path
src_root = Path(__file__).resolve().parents[2]
sys.path.append(str(src_root))
from utils.config import get_project_root

# ---------------------------------------------------------
# 設定：被験者指定
# ---------------------------------------------------------
# ▼▼▼【変更点】被験者IDの指定 ▼▼▼
# 特定の被験者のみ処理したい場合はここにIDを記入（例: "P002"）
# ※ 全員分処理したい場合は None または "" (空文字) にしてください
target_subject_id = ""
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ---------------------------------------------------------
# SGフィルターの設定（60fps用）
# ---------------------------------------------------------
def apply_sg_filter(df, win=21, poly=3):
    """
    df : 読み込んだCSV（CoG_X_mm, CoG_Y_mm を含む）
    win: window length（必ず奇数）
    """
    df = df.copy()

    # 欠損値があっても補間して処理
    x = df["CoG_X_mm"].interpolate().to_numpy()
    y = df["CoG_Y_mm"].interpolate().to_numpy()

    df["CoG_X_mm_sg"] = savgol_filter(x, window_length=win, polyorder=poly)
    df["CoG_Y_mm_sg"] = savgol_filter(y, window_length=win, polyorder=poly)

    return df


# ---------------------------------------------------------
# 1. Base Root
# ---------------------------------------------------------
BASE_ROOT = get_project_root()
INPUT_ROOT = BASE_ROOT / "data" / "2_time_series_feature" / "main_research" / "CoG"
OUTPUT_ROOT = BASE_ROOT / "data"/ "3_summary_feature" / "ROMBERG_ratio"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------------------------------------
# 2. 定義
# ---------------------------------------------------------
GROUPS = ["NOCIPN","CIPN","STUDENT"]
TASKS = ["ROMBERG"]

ROMBERG_CONDS = ["EO", "EC"]
ONELEG_CONDS = ["L", "R"]
TRIALS = ["T1", "T2", "T3"]

FRAME_START = 200
FRAME_END = 1700

pattern = re.compile(
    r"detected_(STUDENT|CIPN|NOCIPN)-(ROMBERG|ONELEG)-(P\d{3})-\d{8}-(EO|EC|L|R)-(T[123])-(C1|C2)_trim_CoG\.csv",
    re.IGNORECASE
)

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def compute_path_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def compute_convex_area(x, y):
    try:
        hull = ConvexHull(np.column_stack((x, y)))
        return hull.volume
    except:
        return np.nan


# ---------------------------------------------------------
# 3. メイン処理
# ---------------------------------------------------------
for group in GROUPS:

    group_dir = os.path.join(INPUT_ROOT, group)
    if not os.path.exists(group_dir):
        print(f"\n✖ Missing group folder: {group_dir}")
        continue

    # 存在する全被験者を取得
    all_subjects = sorted([d for d in os.listdir(group_dir) if d.startswith("P")])

    # ▼▼▼【変更点】リストのフィルタリング ▼▼▼
    if target_subject_id:
        if target_subject_id in all_subjects:
            subjects = [target_subject_id]
            print(f"\n==============================")
            print(f"▶ GROUP = {group} (Target: {target_subject_id})")
            print("==============================")
        else:
            print(f"⚠ {target_subject_id} not found in {group}. Skipping.")
            subjects = []
    else:
        subjects = all_subjects
        print(f"\n==============================")
        print(f"▶ GROUP = {group} (All subjects)")
        print("==============================")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    for sid in subjects:

        print(f"\n--- Subject {sid} ---")
        subject_dir = os.path.join(group_dir, sid)

        for task in TASKS:

            task_dir = os.path.join(subject_dir, task)
            if not os.path.exists(task_dir):
                print(f"  ✖ Missing task folder: {task_dir}")
                continue

            print(f"  ▶ TASK = {task}")

            c1_dir = os.path.join(task_dir, "C1")
            c2_dir = os.path.join(task_dir, "C2")

            if not (os.path.exists(c1_dir) and os.path.exists(c2_dir)):
                print("    ✖ Missing C1 or C2 folder")
                continue

            files_c1 = glob.glob(os.path.join(c1_dir, "*.csv"))
            files_c2 = glob.glob(os.path.join(c2_dir, "*.csv"))

            index_c1, index_c2 = {}, {}

            # C1
            for f in files_c1:
                m = pattern.search(os.path.basename(f))
                if m:
                    g, t, pid, cond, trial, cam = m.groups()
                    if t == task:
                        index_c1[(pid, cond, trial)] = f

            # C2
            for f in files_c2:
                m = pattern.search(os.path.basename(f))
                if m:
                    g, t, pid, cond, trial, cam = m.groups()
                    if t == task:
                        index_c2[(pid, cond, trial)] = f

            # プロット準備
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            fig.suptitle(f"{group} - {sid} - {task}", fontsize=16)

            all_x, all_y = [], []

            CONDITIONS = ROMBERG_CONDS if task == "ROMBERG" else ONELEG_CONDS
            metrics = {c: {"path": [], "area": []} for c in CONDITIONS}

            # 各条件 × 試行
            for i, cond in enumerate(CONDITIONS):
                for j, trial in enumerate(TRIALS):

                    ax = axes[i, j]
                    key = (sid, cond, trial)

                    f1 = index_c1.get(key)
                    f2 = index_c2.get(key)

                    if not f1 or not f2:
                        ax.set_visible(False)
                        continue

                    # --------------------------
                    # SG フィルター適用
                    # --------------------------
                    try:
                        df1 = apply_sg_filter(pd.read_csv(f1), win=21, poly=3)
                        df2 = apply_sg_filter(pd.read_csv(f2), win=21, poly=3)
                    except Exception as e:
                        print(f"    ⚠ Error reading CSV: {e}")
                        ax.set_visible(False)
                        continue

                    n = min(len(df1), len(df2))
                    if n < FRAME_START:
                        ax.set_visible(False)
                        continue

                    # 注意：C1 → x, C2 → y
                    x = df1["CoG_X_mm_sg"].iloc[FRAME_START:n].to_numpy()
                    y = df2["CoG_X_mm_sg"].iloc[FRAME_START:n].to_numpy()

                    all_x.extend(x)
                    all_y.extend(y)

                    path_len = compute_path_length(x, y)
                    area = compute_convex_area(x, y)

                    metrics[cond]["path"].append(path_len)
                    metrics[cond]["area"].append(area)

                    # --------------------------
                    # プロット
                    # --------------------------
                    ax.plot(x, y, lw=1.4, color="royalblue")

                    try:
                        hull = ConvexHull(np.column_stack((x, y)))
                        pts = np.column_stack((x, y))[hull.vertices]
                        ax.fill(pts[:, 0], pts[:, 1], color="salmon", alpha=0.25)
                        ax.plot(pts[:, 0], pts[:, 1], color="tomato", lw=1.2)
                    except:
                        pass

                    ax.text(
                        0.98, 0.92,
                        f"{path_len:.1f} mm\n{area:.1f} mm²",
                        ha="right", va="top",
                        transform=ax.transAxes,
                        fontsize=9,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6)
                    )

                    ax.set_title(f"{cond}-{trial}")
                    ax.grid(True)

            # 軸揃え
            if len(all_x) > 0:
                xmin, xmax = min(all_x), max(all_x)
                ymin, ymax = min(all_y), max(all_y)

                for row in axes:
                    for ax in row:
                        if ax.get_visible():
                            ax.set_xlim(xmin, xmax)
                            ax.set_ylim(ymin, ymax)
                            ax.set_aspect("equal", "box")

            # ---------------------------------------------------------
            # プロットに Romberg Ratio を書く（ROMBERG のみ）
            # ---------------------------------------------------------
            if task == "ROMBERG":
                EO_path = np.nanmean(metrics["EO"]["path"])
                EC_path = np.nanmean(metrics["EC"]["path"])
                EO_area = np.nanmean(metrics["EO"]["area"])
                EC_area = np.nanmean(metrics["EC"]["area"])

                RR_path = EC_path / EO_path if EO_path > 0 else np.nan
                RR_area = EC_area / EO_area if EO_area > 0 else np.nan

                fig.text(
                    0.98, 0.02,
                    f"RR_path = {RR_path:.2f}\nRR_area = {RR_area:.2f}",
                    ha="right", va="bottom",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
                )

            # ---------------------------------------------------------
            # 保存
            # ---------------------------------------------------------
            plot_dir = os.path.join(OUTPUT_ROOT, group, task, "filtered_plots")
            os.makedirs(plot_dir, exist_ok=True)
            out_path = os.path.join(plot_dir, f"{sid}_{task}_filtered.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"    ✓ Plot saved: {out_path}")

            # ---------------------------------------------------------
            # summary（条件別平均 + ロンベルク率）
            # ---------------------------------------------------------
            summary_dir = os.path.join(OUTPUT_ROOT, group, task, "filtered_summary")
            os.makedirs(summary_dir, exist_ok=True)

            row = {"subject": sid}
            for c in CONDITIONS:
                row[f"{c}_path"] = np.nanmean(metrics[c]["path"])
                row[f"{c}_area"] = np.nanmean(metrics[c]["area"])

            # ---- ロンベルク率（ROMBERGのみ） ----
            if task == "ROMBERG":
                EO_p = row["EO_path"]
                EC_p = row["EC_path"]
                EO_a = row["EO_area"]
                EC_a = row["EC_area"]

                row["RR_path"] = EC_p / EO_p if EO_p > 0 else np.nan
                row["RR_area"] = EC_a / EO_a if EO_a > 0 else np.nan
            else:
                row["RR_path"] = np.nan
                row["RR_area"] = np.nan

            summary_path = os.path.join(summary_dir, f"{group}_{task}_summary.csv")

            # ▼▼▼【変更点】CSVへの更新ロジック ▼▼▼
            df_new_row = pd.DataFrame([row])

            if os.path.exists(summary_path):
                try:
                    df_master = pd.read_csv(summary_path)
                    
                    # 既存データから、今回計算した subject の行を削除（更新のため）
                    before_len = len(df_master)
                    df_master = df_master[df_master["subject"] != sid]
                    
                    if len(df_master) != before_len:
                        print(f"    ↻ Updating existing entry in CSV.")
                    else:
                        print(f"    + Appending new entry to CSV.")
                        
                    # 結合
                    df_final = pd.concat([df_master, df_new_row], ignore_index=True)
                except Exception as e:
                    print(f"    ⚠ Error reading existing CSV. Overwriting. ({e})")
                    df_final = df_new_row
            else:
                print(f"    ★ Creating new summary CSV.")
                df_final = df_new_row

            # subjectでソートして保存
            df_final = df_final.sort_values("subject")
            df_final.to_csv(summary_path, index=False)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

print("\n=== All Processing Complete ===")