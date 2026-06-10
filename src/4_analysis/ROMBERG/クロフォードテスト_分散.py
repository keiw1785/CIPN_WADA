import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import sys
from pathlib import Path
src_root = Path(__file__).resolve().parents[2]
sys.path.append(str(src_root))
from utils.config import get_project_root


# =========================================================
# 1. 設定 (SETTINGS)
# =========================================================
BASE_ROOT = get_project_root()

# 分散の集計CSVのパス
VARIANCE_SUMMARY_FILE = BASE_ROOT / "data" / "3_summary_feature" / "ROMBERG_Variance" / "summary" / "ROMBERG_AllVariance_summary.csv"

# 結果出力先
OUTPUT_DIR = BASE_ROOT / "daily_results" / "20260528" / "分散" / "Crawford_Variance_Full_Unique"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_GROUPS = ["CIPN", "NOCIPN"] 
CONTROL_GROUP = "STUDENT"
GROUP_FOR_COMPARISON = "CIPN" # 群間比較用の代表グループ

# ▼▼▼ フォント・スタイル設定 ▼▼▼
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['font.size'] = 30
sns.set_theme(style="whitegrid", rc={"font.family": "Arial"})

# =========================================================
# 2. 関数定義
# =========================================================
def perform_crawford_test(case_val, control_vals):
    n = len(control_vals)
    mean_c = np.mean(control_vals)
    std_c = np.std(control_vals, ddof=1)
    if std_c == 0: return np.nan, np.nan, mean_c, std_c
    t_val = (case_val - mean_c) / (std_c * np.sqrt((n + 1) / n))
    df = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_val), df))
    return t_val, p_val, mean_c, std_c

def format_p_value(p):
    if pd.isna(p): return "-"
    if p < 0.001: return "< 0.001*"
    elif p < 0.05: return f"{p:.3f}*"
    else: return f"{p:.3f}"

# =========================================================
# 3. データ読み込み
# =========================================================
print(f"--- Loading Variance Data ---")
if not os.path.exists(VARIANCE_SUMMARY_FILE):
    raise FileNotFoundError(f"CSVが見つかりません: {VARIANCE_SUMMARY_FILE}")
df_all = pd.read_csv(VARIANCE_SUMMARY_FILE)

# =========================================================
# 4. カメラごとのループ処理 (C1, C2)
# =========================================================
CAMERAS = ["C1", "C2"]

for cam in CAMERAS:
    print(f"\n{'='*20} Processing Camera: {cam} {'='*20}")

    # 1. コントロール群
    df_control_cam = df_all[(df_all["Group"] == CONTROL_GROUP) & (df_all["camera"] == cam)].copy()
    
    # 2. ターゲット群
    df_target_cam = df_all[(df_all["Group"].isin(TARGET_GROUPS)) & (df_all["camera"] == cam)].copy()

    if len(df_control_cam) < 3:
        print(f"⚠ Skipping {cam}: Not enough control data.")
        continue

    # ▼▼▼ 被験者IDのユニーク化 (追加箇所) ▼▼▼
    df_target_cam["subject_display"] = df_target_cam["subject"] + " (" + df_target_cam["Group"] + ")"
    df_target_cam = df_target_cam.sort_values(by=["Group", "subject"])
    target_list = df_target_cam["subject_display"].unique().tolist()
    
    df_cipn_group = df_target_cam[df_target_cam["Group"] == GROUP_FOR_COMPARISON].copy()

    # -----------------------------------------------------
    # 統計解析 (Crawford Test - Individual)
    # -----------------------------------------------------
    metrics_all = ["Local_EO", "Local_EC", "Local_RR", "Global_EO", "Global_EC", "Global_RR", "Total_EO", "Total_EC", "Total_RR"]
    summary_results = []

    for pid_disp in target_list:
        patient_data = df_target_cam[df_target_cam["subject_display"] == pid_disp].iloc[0]
        for metric in metrics_all:
            val_p = patient_data[metric]
            vals_h = df_control_cam[metric].dropna()
            _, p_val, _, _ = perform_crawford_test(val_p, vals_h)
            summary_results.append({
                "Camera": cam, "Metric": metric, "Patient_ID": pid_disp, "Group": patient_data["Group"],
                "Patient_Value": val_p, "p_value": p_val, "Significant": "YES" if p_val < 0.05 else "no"
            })

    df_stats = pd.DataFrame(summary_results)
    df_stats.to_csv(os.path.join(OUTPUT_DIR, f"Stats_{cam}.csv"), index=False)

    # -----------------------------------------------------
    # 図の作成 & 表の作成 (カテゴリごと)
    # -----------------------------------------------------
    plot_categories = {
        "Local_Variance":  ["Local_EO", "Local_EC", "Local_RR"],
        "Global_Variance": ["Global_EO", "Global_EC", "Global_RR"],
        "Total_Variance":  ["Total_EO", "Total_EC", "Total_RR"]
    }

    label_dict = {
        "EO": "EO Variance", "EC": "EC Variance", "RR": "Romberg Ratio"
    }

    for cat_name, metrics in plot_categories.items():
        # --- A. 図の作成 ---
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            # Control Box
            sns.boxplot(y=df_control_cam[metric], width=0.6, ax=ax, 
                        boxprops=dict(facecolor="#BDFFA3", edgecolor="#888888", alpha=0.6, linewidth=2),
                        showfliers=False, zorder=1)
            # CIPN Box
            if len(df_cipn_group) > 0:
                sns.boxplot(y=df_cipn_group[metric], width=0.3, ax=ax,
                            boxprops=dict(facecolor="#FFCCCC", edgecolor="red", alpha=0.7, linewidth=3),
                            showfliers=False, zorder=2)

            # 散布図プロット
            np.random.seed(42)
            for pid in target_list:
                p_data = df_target_cam[df_target_cam["subject_display"] == pid].iloc[0]
                val_p = p_data[metric]
                is_sig = df_stats[(df_stats["Patient_ID"] == pid) & (df_stats["Metric"] == metric)]["Significant"].values[0] == "YES"
                
                p_color = "red" if "(CIPN)" in pid else "blue"
                marker, size, edge, lw = ('D', 1000, 'black', 2.5) if is_sig else ('o', 800, 'white', 2.0)
                
                ax.scatter(np.random.uniform(-0.08, 0.08), val_p, color=p_color, marker=marker, s=size, 
                           edgecolors=edge, linewidth=lw, alpha=0.9, zorder=10)

            # 軸ラベル
            m_type = metric.split("_")[1]
            ax.set_title(f"{cat_name.split('_')[0]} {label_dict[m_type]}", fontsize=35, fontweight="bold", pad=20)
            ax.set_ylabel("Romberg Ratio" if "RR" in metric else "Variance (mm²)", fontsize=30)
            ax.set_xlabel(f"Box: Green=Control, Red={GROUP_FOR_COMPARISON}", fontsize=22)
            ax.set_xticks([]); sns.despine(ax=ax, left=False, bottom=True)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        # 凡例
        custom_lines = [
            Line2D([0], [0], color="#BDFFA3", lw=10, label='Control Box'),
            Line2D([0], [0], color='#FFCCCC', lw=10, label=f'{GROUP_FOR_COMPARISON} Box'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=20, label='CIPN'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=20, label='Non-CIPN'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=20, label='Sig (p<0.05)')
        ]
        fig.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=24, title="Legend", title_fontsize=26)
        
        plt.tight_layout(); plt.subplots_adjust(right=0.95)
        plt.savefig(os.path.join(OUTPUT_DIR, f"Plot_{cat_name}_{cam}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # --- B. 表の作成 ---
        col_labels = ["Subject"]
        for m in metrics: col_labels.extend([m.split("_")[1], "p"])
        
        table_data = []
        # 1. Control
        row_ctrl = [f"Control (N={len(df_control_cam)})"]
        for m in metrics: row_ctrl.extend([f"{df_control_cam[m].mean():.2f}±{df_control_cam[m].std():.2f}", "-"] )
        table_data.append(row_ctrl)

        # 2. CIPN Mean
        row_cipn_m = [f"{GROUP_FOR_COMPARISON} Mean"]
        for m in metrics:
            mu_c = df_cipn_group[m].mean()
            _, p_g = stats.ttest_ind(df_cipn_group[m].dropna(), df_control_cam[m].dropna(), equal_var=False)
            row_cipn_m.extend([f"{mu_c:.2f}", format_p_value(p_g)])
        table_data.append(row_cipn_m)

        # 3. Individual
        for pid in target_list:
            row = [pid]
            for m in metrics:
                res = df_stats[(df_stats["Patient_ID"] == pid) & (df_stats["Metric"] == m)]
                row.extend([f"{res['Patient_Value'].values[0]:.2f}", format_p_value(res['p_value'].values[0])])
            table_data.append(row)

        # 描画設定
        fig_tbl, ax_tbl = plt.subplots(figsize=(24, len(table_data)*0.8 + 2))
        ax_tbl.axis('off')
        table = ax_tbl.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colWidths=[0.3] + [0.12, 0.08]*3)
        table.auto_set_font_size(False); table.set_fontsize(24); table.scale(1, 2.5)

        # 表の色付け
        for (r, c), cell in table.get_celld().items():
            if r == 0: cell.set_facecolor("#f0f0f0") # Header
            elif r == 1: cell.set_facecolor("#f9f9f9") # Control
            elif r == 2: cell.set_facecolor("#ffcccc") # Group Mean
            if c == 0 and r > 2:
                if "(CIPN)" in table_data[r-1][0]: cell.get_text().set_color("darkred")
                elif "(NOCIPN)" in table_data[r-1][0]: cell.get_text().set_color("navy")

        plt.savefig(os.path.join(OUTPUT_DIR, f"Table_{cat_name}_{cam}.png"), dpi=300, bbox_inches='tight')
        plt.close()

print("\n=== Analysis Complete ===")