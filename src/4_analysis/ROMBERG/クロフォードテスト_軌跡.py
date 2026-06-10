import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.lines import Line2D

# =========================================================
# 1. 設定 (SETTINGS)
# =========================================================
BASE_ROOT = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA"
SUMMARY_ROOT = os.path.join(BASE_ROOT, r"data/3_summary_feature/ROMBERG_ratio")

# 出力先
OUTPUT_DIR = os.path.join(BASE_ROOT, r"daily_results/20260528/Crawford_Romberg_LargeStyle")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 解析対象
TARGET_GROUPS = ["CIPN", "NOCIPN"] 
CONTROL_GROUP = "STUDENT" 
GROUP_FOR_COMPARISON = "CIPN" # 箱ひげ図と表で強調表示するグループ
TARGET_IDS = "ALL"

# ▼▼▼ スタイル設定 (巨大化) ▼▼▼
plt.rcParams['font.family'] = 'Arial' # 日本語が必要なら 'MS Gothic'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 24

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": ".8", 
    "grid.color": ".9",
    "font.family": "Arial",
    "font.size": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30
})

# =========================================================
# 2. 関数定義
# =========================================================
def load_summary_csv(group_name):
    csv_path = os.path.join(
        SUMMARY_ROOT, group_name, "ROMBERG", "filtered_summary", f"{group_name}_ROMBERG_summary.csv"
    )
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["Group_Label"] = group_name
    df["subject"] = df["subject"].astype(str) + " (" + group_name + ")"
    return df

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
print(f"--- Loading Data ---")
df_control = load_summary_csv(CONTROL_GROUP)
if df_control is None: raise FileNotFoundError("Control missing")

target_dfs = []
for grp in TARGET_GROUPS:
    df_temp = load_summary_csv(grp)
    if df_temp is not None:
        target_dfs.append(df_temp)

if not target_dfs: raise FileNotFoundError("Target missing")
df_target_all = pd.concat(target_dfs, ignore_index=True)
target_list = df_target_all["subject"].unique().tolist()
print(f"Targets: {len(target_list)} subjects")

# 群間比較用のデータ準備
df_cipn_group = df_target_all[df_target_all["Group_Label"] == GROUP_FOR_COMPARISON].copy()
df_nocipn_group = df_target_all[df_target_all["Group_Label"] == "NOCIPN"].copy() # ★追加箇所

# =========================================================
# 4. 統計解析
# =========================================================
metrics_list = ["EO_path", "EO_area", "EC_path", "EC_area", "RR_path", "RR_area"]
summary_results = []
print(f"--- Performing Analysis ---")

for pid in target_list:
    row = df_target_all[df_target_all["subject"] == pid].iloc[0]
    p_group = row["Group_Label"]
    
    for metric in metrics_list:
        val_p = row[metric]
        vals_h = df_control[metric].dropna()
        
        # 本人がコントロール群に含まれる場合は除外（念のため）
        if p_group == CONTROL_GROUP:
            vals_h = vals_h[df_control["subject"] != pid]

        t_stat, p_val, mean_h, std_h = perform_crawford_test(val_p, vals_h)

        summary_results.append({
            "Metric": metric,
            "Patient_ID": pid,
            "Group": p_group,
            "Patient_Value": val_p,
            "p_value": p_val,
            "Significant": "YES" if p_val < 0.05 else "no"
        })

df_stats = pd.DataFrame(summary_results)
# =========================================================
# CIPNで有意ではない被験者名だけを表示
# =========================================================
cipn_non_sig_subjects = df_stats[
    (df_stats["Group"] == "CIPN") &
    (df_stats["Metric"] == "RR_path") &
    (df_stats["Significant"] == "no")
]["Patient_ID"].unique()

print("\n--- CIPNで有意ではない被験者 ---")
for sid in cipn_non_sig_subjects:
    print(sid)

csv_out_path = os.path.join(OUTPUT_DIR, "Stats_Romberg_Large.csv")
df_stats.to_csv(csv_out_path, index=False)
csv_out_path = os.path.join(OUTPUT_DIR, "Stats_Romberg_Large.csv")
df_stats.to_csv(csv_out_path, index=False)

# =========================================================
# 5. 色の設定
# =========================================================
cipn_subjects = [s for s in target_list if "(CIPN)" in s]
nocipn_subjects = [s for s in target_list if "(NOCIPN)" in s]

color_cipn_fixed = "red"
color_nocipn_fixed = "blue"

color_dict = {}
for sub in cipn_subjects: 
    color_dict[sub] = color_cipn_fixed
for sub in nocipn_subjects: 
    color_dict[sub] = color_nocipn_fixed

# =========================================================
# 6. 図の作成
# =========================================================
plot_groups = {
    "Path_Analysis": [
        ("EO_path", "Total EO Path"), 
        ("EC_path", "Total EC Path"), 
        ("RR_path", "Total Romberg Ratio (Path)")
    ],
    "Area_Analysis": [
        ("EO_area", "Total EO Area"), 
        ("EC_area", "Total EC Area"), 
        ("RR_area", "Total Romberg Ratio (Area)")
    ]
}

for group_name_plot, metrics_in_group in plot_groups.items():
    fig, axes = plt.subplots(1, 3, figsize=(32, 10)) 
    
    for i, (metric, label) in enumerate(metrics_in_group):
        ax = axes[i]
        
        # --- 箱ひげ図 1: Control群 ---
        vals_h = df_control[metric].dropna()
        sns.boxplot(y=vals_h, width=0.5, ax=ax, 
                    boxprops=dict(facecolor="#BDFFA3", edgecolor="#888888", alpha=0.6, linewidth=2),
                    showfliers=False, zorder=1)
        
        # --- 箱ひげ図 2: CIPN群 ---
        vals_cipn = df_cipn_group[metric].dropna()
        if len(vals_cipn) > 0:
            sns.boxplot(y=vals_cipn, width=0.25, ax=ax,
                        boxprops=dict(facecolor="#FFCCCC", edgecolor="red", alpha=0.7, linewidth=3),
                        showfliers=False, zorder=2)

        # --- 散布図プロット ---
        np.random.seed(42)
        for pid in target_list:
            row = df_target_all[df_target_all["subject"] == pid].iloc[0]
            val_p = row[metric]
            p_color = color_dict.get(pid, "black")
            
            stat_row = df_stats[(df_stats["Patient_ID"] == pid) & (df_stats["Metric"] == metric)]
            is_sig = stat_row["Significant"].values[0] == "YES" if not stat_row.empty else False
            
            if is_sig:
                marker, size, edge_c, lw, z_ord = ('D', 1000, 'black', 2.5, 20)
            else:
                marker, size, edge_c, lw, z_ord = ('o', 800, 'white', 2.0, 10)
            
            x_pos = np.random.uniform(-0.06, 0.06)
            ax.scatter(x_pos, val_p, color=p_color, marker=marker, s=size, zorder=z_ord,
                        edgecolors=edge_c, linewidth=lw, alpha=0.9)

        # 軸設定
        ax.set_title(label, fontsize=35, fontweight="bold", pad=20)
        
        if "Ratio" in label: ylabel = "Romberg Ratio"
        elif "Area" in label: ylabel = "Variance (mm²)"
        else: ylabel = "Path Length (mm)"
            
        ax.set_ylabel(ylabel, fontsize=30)
        ax.set_xlabel(f"Box: Green=Control, Red={GROUP_FOR_COMPARISON}", color="#333333", fontsize=20)
        ax.set_xticks([])
        sns.despine(ax=ax, left=False, bottom=True, top=True, right=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 凡例設定
    custom_lines = [
        Line2D([0], [0], color="#BDFFA3", lw=10, label='Control Box'),
        Line2D([0], [0], color='#FFCCCC', lw=10, label=f'{GROUP_FOR_COMPARISON} Box'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_cipn_fixed, markeredgecolor='w', markersize=20, label='CIPN'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_nocipn_fixed, markeredgecolor='w', markersize=20, label='Non-CIPN'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='white', markersize=20, label='Non-Sig'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=20, label='Sig (p<0.05)')
    ]

    fig.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                title="Legend", frameon=True, fancybox=True, borderpad=1, fontsize=24, title_fontsize=26)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)
    
    img_path = os.path.join(OUTPUT_DIR, f"Combined_3panels_{group_name_plot}.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {os.path.basename(img_path)}")

# =========================================================
# 7. 表の作成と出力 (巨大スタイル適用)
# =========================================================
print("--- Generating Summary Table (Large Style) ---")
plt.rcParams['font.family'] = 'MS Gothic' # 日本語フォント

metrics_for_table = ["EO_path", "EO_area", "EC_path", "EC_area", "RR_path", "RR_area"]

col_labels = ["Subject"]
for m in metrics_for_table:
    cond, tipo = m.split("_")
    short_name = f"{cond} {tipo[0].upper()}" 
    col_labels.extend([short_name, "p"])

table_data = []

# === 1. コントロール群 (統計) ===
row_ctrl = [f"Control (N={len(df_control)})"]
for m in metrics_for_table:
    vals = df_control[m].dropna()
    mu, sigma = np.mean(vals), np.std(vals, ddof=1)
    row_ctrl.extend([f"{mu:.2f}±{sigma:.2f}", "-"])
table_data.append(row_ctrl)

# === 2. CIPN群 (統計) & 群間比較 ===
row_cipn_stats = [f"{GROUP_FOR_COMPARISON} Mean (N={len(df_cipn_group)})"]
for m in metrics_for_table:
    vals_ctrl = df_control[m].dropna()
    vals_cipn = df_cipn_group[m].dropna()
    
    if len(vals_cipn) > 1 and len(vals_ctrl) > 1:
        mu_c = np.mean(vals_cipn)
        std_c = np.std(vals_cipn, ddof=1)
        t_stat_g, p_val_g = stats.ttest_ind(vals_cipn, vals_ctrl, equal_var=False)
        row_cipn_stats.extend([f"{mu_c:.2f}±{std_c:.2f}", format_p_value(p_val_g)])
    else:
        row_cipn_stats.extend(["-", "-"])
table_data.append(row_cipn_stats)

# === 3. NOCIPN群 (統計) & 群間比較 (★追加箇所) ===
if len(df_nocipn_group) > 0:
    row_nocipn_stats = [f"NOCIPN Mean (N={len(df_nocipn_group)})"]
    for m in metrics_for_table:
        vals_ctrl = df_control[m].dropna()
        vals_nocipn = df_nocipn_group[m].dropna()
        
        if len(vals_nocipn) > 1 and len(vals_ctrl) > 1:
            mu_n = np.mean(vals_nocipn)
            std_n = np.std(vals_nocipn, ddof=1)
            t_stat_n, p_val_n = stats.ttest_ind(vals_nocipn, vals_ctrl, equal_var=False)
            row_nocipn_stats.extend([f"{mu_n:.2f}±{std_n:.2f}", format_p_value(p_val_n)])
        else:
            row_nocipn_stats.extend(["-", "-"])
    table_data.append(row_nocipn_stats)

# === 4. 個別症例 (Crawford) ===
for pid in target_list:
    row = [pid]
    for m in metrics_for_table:
        stat_row = df_stats[(df_stats["Patient_ID"] == pid) & (df_stats["Metric"] == m)]
        if not stat_row.empty:
            val = stat_row["Patient_Value"].values[0]
            pval = stat_row["p_value"].values[0]
            row.extend([f"{val:.2f}", format_p_value(pval)])
        else:
            row.extend(["-", "-"])
    table_data.append(row)

# --- 表の描画設定 ---
h_cell = 0.8
w_img = 30 
h_img = (len(table_data) + 3) * h_cell

fig_tbl, ax_tbl = plt.subplots(figsize=(w_img, h_img))
ax_tbl.axis('off')

col_widths = [0.2] + [0.06] * 12

table = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
    colWidths=col_widths
)

table.auto_set_font_size(False)
table.set_fontsize(24)
table.scale(1, 2.5) 

header_colors = {
    "EO": "#dae8fc", 
    "EC": "#ffe6cc", 
    "RR": "#d5e8d4"  
}

# ▼▼▼ 表の色付けロジックの修正 ▼▼▼
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', fontsize=26)
        cell.set_linewidth(1.5)
        if col == 0:
            cell.set_facecolor("#f0f0f0")
        elif 1 <= col <= 4:
            cell.set_facecolor(header_colors["EO"])
        elif 5 <= col <= 8:
            cell.set_facecolor(header_colors["EC"])
        elif 9 <= col <= 12:
            cell.set_facecolor(header_colors["RR"])
            
    else:
        cell.set_text_props(fontsize=24)
        row_text = table_data[row-1][0]
        
        # 背景色の設定
        if "Control" in row_text:
            cell.set_facecolor("#f9f9f9")
            if col == 0: cell.set_text_props(weight='bold')
        elif "CIPN Mean" in row_text:
            cell.set_facecolor("#ffcccc")
            if col == 0: cell.set_text_props(weight='bold', color='darkred', fontsize=24)
        elif "NOCIPN Mean" in row_text:
            cell.set_facecolor("#cce5ff") # NOCIPN群の背景: 薄い青
            if col == 0: cell.set_text_props(weight='bold', color='navy', fontsize=24)

        # 個別データの文字色
        elif col == 0:
            if "(CIPN)" in row_text:
                cell.set_text_props(color="darkred")
            elif "(NOCIPN)" in row_text:
                cell.set_text_props(color="navy")

plt.title("Romberg Summary Table (Large Style)", y=1.0, pad=40, fontsize=30, fontweight='bold')

table_img_path = os.path.join(OUTPUT_DIR, "Romberg_Summary_Table_Large.png")
plt.savefig(table_img_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f"Table saved: {os.path.basename(table_img_path)}")