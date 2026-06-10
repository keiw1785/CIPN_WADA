import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# ==========================================
# ★ Settings Area (English Version) ★
# ==========================================
INPUT_CSV_PATH = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results\20260528\TUG\tug_final_ratio_metric\final_svg\tug_metrics_averaged.csv"
INPUT_DIR = os.path.dirname(INPUT_CSV_PATH)
OUTPUT_BASE_DIR = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results\20260528\advanced_analysis_results_EN_RadarOnly_v3" # v3に変更

# Font settings
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
sns.set(font=['Arial', 'DejaVu Sans', 'sans-serif'])

FONT_SIZE_LABEL = 38
FONT_SIZE_TICK = 18

# Mapping: CSV Header -> English Display Label
# ★工夫: 長いラベルは途中で改行(\n)を入れると重なりにくくなります
METRIC_EN_MAP = {
    'TUG Score (s)': 'Total\nDuration (s)',
    'Total Steps': 'Total\nSteps',
    'Turn Radius (m)': 'Turn\nRadius (m)',
    'Turn Duration (s)': 'Turn\nDuration (s)',
    'Stride Length Ratio': 'Stride Ratio\n(Max/Min)',
    'Rise Duration (s)': 'Rise\nTime (s)',
    'Sit Duration (s)': 'Sit\nTime (s)'
}
TARGET_METRICS = list(METRIC_EN_MAP.keys())

# ==========================================
# Visualization Function (Fixed Overlap)
# ==========================================
def plot_radar_chart_raw_mean(df_cond, target_cols, condition_name, output_dir):
    groups = ['Student', 'NOCIPN', 'CIPN']
    df_filtered = df_cond[df_cond['Group'].isin(groups)]
    if len(df_filtered) == 0: return

    # Calculate Mean
    df_means = df_filtered.groupby('Group')[target_cols].mean()

    # Data Processing
    if condition_name == 'RATIO':
        df_plot = df_means.copy()
        is_ratio = True
    else:
        df_plot = df_means.copy()
        max_vals = df_means.max(axis=0)
        for col in target_cols:
            if max_vals[col] != 0:
                df_plot[col] = df_means[col] / max_vals[col]
            else:
                df_plot[col] = 0
        is_ratio = False

    # Preparation
    categories = [METRIC_EN_MAP.get(c, c) for c in target_cols]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 図のサイズ設定
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    # =================================================================
    # ★修正箇所1: ラベルの位置調整 (Padding)
    # =================================================================
    # まず目盛り位置を設定
    ax.set_xticks(angles[:-1])
    
    # ラベルを設定
    ax.set_xticklabels(categories, color='black', size=FONT_SIZE_LABEL)
    
    # ★重要: pad=60 でグラフから文字を離します (フォント32の場合はこれくらい必要)
    ax.tick_params(axis='x', pad=60)

    # Y-Axis Settings
    ax.set_rlabel_position(0)
    
    if is_ratio:
        data_max = df_plot.max().max()
        limit = np.ceil(data_max * 10) / 10 + 0.1
        if limit < 1.2: limit = 1.2
        
        ticks = [0.5, 1.0, limit]
        tick_labels = ["0.5x", "1.0x", f"{limit:.1f}x"]
        plt.yticks(ticks, tick_labels, color="grey", size=FONT_SIZE_TICK)
        plt.ylim(0, limit)
        ax.plot(angles, [1.0]*(N+1), 'k--', linewidth=1, alpha=0.5)
        
    else:
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], 
                   color="grey", size=FONT_SIZE_TICK)
        plt.ylim(0, 1.1)

    # Plot Settings
    colors = {'Student': 'green', 'NOCIPN': 'blue', 'CIPN': 'red'}
    styles = {'Student': '--', 'NOCIPN': '-', 'CIPN': '-'}
    widths = {'Student': 2, 'NOCIPN': 3, 'CIPN': 4}

    for grp in groups:
        if grp in df_plot.index:
            values = df_plot.loc[grp].values.flatten().tolist()
            values += values[:1]
            
            ax.plot(angles, values, 
                    linewidth=widths.get(grp, 2), 
                    linestyle=styles.get(grp, '-'), 
                    color=colors.get(grp, 'black'),
                    label=grp)
            
            ax.fill(angles, values, colors.get(grp, 'black'), alpha=0.05)

    # Legend Settings
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
               fontsize=20, title="Group", title_fontsize=24, frameon=True)
    
    # 余白調整
    plt.tight_layout()
    
    suffix = "RawRatio" if is_ratio else "Scaled"
    
    # Save
    save_path_svg = os.path.join(output_dir, f"Radar_{suffix}_{condition_name}.svg")
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    
    save_path_png = os.path.join(output_dir, f"Radar_{suffix}_{condition_name}.png")
    plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"Saved: {os.path.basename(save_path_svg)}")

# ==========================================
# Main Execution
# ==========================================
def main():
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found.")
        return
    
    print(f"Loading data...")
    df_mean = pd.read_csv(INPUT_CSV_PATH)

    if 'RATIO' in df_mean['Condition'].unique():
        df_all = df_mean
    else:
        print("Calculating Ratio...")
        missing_cols = [c for c in TARGET_METRICS if c not in df_mean.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            return

        df_pivot = df_mean.pivot(index=['Group', 'Subject'], columns='Condition', values=TARGET_METRICS)
        df_ratio = pd.DataFrame(index=df_pivot.index)
        valid_ratio = False
        for col in TARGET_METRICS:
            if 'MAX' in df_pivot[col].columns and 'NORMAL' in df_pivot[col].columns:
                df_ratio[col] = df_pivot[col]['MAX'] / df_pivot[col]['NORMAL']
                valid_ratio = True
        
        if valid_ratio:
            df_ratio = df_ratio.reset_index()
            df_ratio['Condition'] = 'RATIO'
            df_all = pd.concat([df_mean, df_ratio], ignore_index=True)
        else:
            df_all = df_mean

    for cond in ['NORMAL', 'MAX', 'RATIO']:
        print(f"--- Analysis: {cond} ---")
        df_cond = df_all[df_all['Condition'] == cond]
        if len(df_cond) == 0: continue
        
        plot_radar_chart_raw_mean(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)

    print(f"\nCompleted: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()