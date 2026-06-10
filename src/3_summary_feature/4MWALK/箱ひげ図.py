import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # 画面描画をオフにして高速化＆フリーズ防止
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats

# ==========================================
# Settings: Output and Input Paths
# ==========================================
OUTPUT_BASE_DIR = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results\20260528\4M\walk_analysis_normalized_ratio_with_stats"
INPUT_ROOT_DIR = r'C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\3D_Result' 

plt.rcParams['font.family'] = 'MS Gothic'

# ==========================================
# 1. Common Functions
# ==========================================
def extract_file_info(filename):
    fname_upper = filename.upper()
    if "NOCIPN" in fname_upper: group = "NOCIPN"
    elif "CIPN" in fname_upper: group = "CIPN"
    elif "STUDENT" in fname_upper: group = "Student"
    else: group = "Unknown"
    
    match = re.search(r'(P\d+)', fname_upper)
    subject_id = match.group(1) if match else "Unknown"
    
    if "MAX" in fname_upper: condition = "MAX"
    elif "NORMAL" in fname_upper: condition = "NORMAL"
    else: condition = "Unknown"
        
    return group, subject_id, condition

def lowpass_filter(data, fps, cutoff=6.0, order=4):
    nyq = 0.5 * fps
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ==========================================
# 2. 4m Walk Analysis Logic
# ==========================================
def analyze_center_4m_walk(file_path):
    try:
        df = pd.read_csv(file_path).dropna()
        if len(df) < 10: return None
        
        if 'TIME' in df.columns:
            time = df['TIME'].values / 1000.0
        else:
            time = np.arange(len(df)) / 30.0
        time = time - time[0]
        dt = np.median(np.diff(time))
        fps = 1.0 / dt if dt > 0 else 30.0

        cols = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']
        axes = ['X', 'Y', 'Z']
        
        ranges = {}
        for ax in axes:
            vals = df[f'left_hip_{ax}'].values
            ranges[ax] = np.max(vals) - np.min(vals)
        fwd_axis = max(ranges, key=ranges.get)
        
        h_s_diffs = {}
        for ax in axes:
            if ax == fwd_axis: continue
            h = (df[f'left_hip_{ax}'] + df[f'right_hip_{ax}']) / 2
            s = (df[f'left_shoulder_{ax}'] + df[f'right_shoulder_{ax}']) / 2
            h_s_diffs[ax] = np.mean(np.abs(s - h))
        vert_axis = max(h_s_diffs, key=h_s_diffs.get)
        lat_axis = [ax for ax in axes if ax not in [fwd_axis, vert_axis]][0]

        l_fwd_raw = df[f'left_ankle_{fwd_axis}'].values
        r_fwd_raw = df[f'right_ankle_{fwd_axis}'].values
        ank_fwd = (l_fwd_raw + r_fwd_raw) / 2 
        
        min_pos = np.min(ank_fwd); max_pos = np.max(ank_fwd)
        total_dist = max_pos - min_pos
        center_pos = (max_pos + min_pos) / 2
        
        TARGET_DIST = 4.0; HALF_DIST = TARGET_DIST / 2.0
        
        if total_dist < TARGET_DIST:
            mask = np.ones(len(df), dtype=bool)
        else:
            cut_min = center_pos - HALF_DIST
            cut_max = center_pos + HALF_DIST
            mask = (ank_fwd >= cut_min) & (ank_fwd <= cut_max)

        if np.sum(mask) < 10: return None
            
        df_ss = df[mask].reset_index(drop=True)
        time_ss = time[mask]; time_ss = time_ss - time_ss[0]
        
        l_fwd = lowpass_filter(df_ss[f'left_ankle_{fwd_axis}'].values, fps)
        r_fwd = lowpass_filter(df_ss[f'right_ankle_{fwd_axis}'].values, fps)
        l_lat = lowpass_filter(df_ss[f'left_ankle_{lat_axis}'].values, fps)
        r_lat = lowpass_filter(df_ss[f'right_ankle_{lat_axis}'].values, fps)

        analyzed_dist = np.abs(ank_fwd[mask][-1] - ank_fwd[mask][0])
        duration = time_ss[-1]
        speed = analyzed_dist / duration if duration > 0 else 0
        
        dist_ankles = np.sqrt(
            (df_ss[f'left_ankle_{fwd_axis}'] - df_ss[f'right_ankle_{fwd_axis}'])**2 +
            (df_ss[f'left_ankle_{lat_axis}'] - df_ss[f'right_ankle_{lat_axis}'])**2 +
            (df_ss[f'left_ankle_{vert_axis}'] - df_ss[f'right_ankle_{vert_axis}'])**2
        )
        dist_ankles = lowpass_filter(dist_ankles.values, fps, cutoff=4.0)
        peaks, _ = find_peaks(dist_ankles, height=0.1, distance=int(fps*0.3))
        
        stride_lengths = []
        step_widths = []
        for p in peaks:
            sl = np.abs(l_fwd[p] - r_fwd[p]) * 100
            stride_lengths.append(sl)
            sw = np.abs(l_lat[p] - r_lat[p]) * 100
            step_widths.append(sw)
            
        avg_stride = np.mean(stride_lengths) if len(stride_lengths) > 0 else 0
        avg_width = np.mean(step_widths) if len(step_widths) > 0 else 0
        
        step_count = len(peaks)
        cadence = (step_count / duration) * 60 if duration > 0 else 0
        
        metrics = {
            "Gait Speed (m/s)": round(speed, 2),
            "Stride Length (cm)": round(avg_stride, 2),
            "Step Width (cm)": round(avg_width, 2),
            "Cadence (steps/min)": round(cadence, 1)
        }
        return metrics

    except Exception:
        return None

# ==========================================
# 3. Ratio Calculation Logic
# ==========================================
def calculate_max_normal_ratios(df_results, target_cols):
    ratio_results = []
    subjects_info = df_results[['Group', 'Subject ID']].drop_duplicates()
    
    for _, row in subjects_info.iterrows():
        group = row['Group']
        sub = row['Subject ID']
        
        sub_df = df_results[(df_results['Group'] == group) & (df_results['Subject ID'] == sub)]
        if sub_df.empty: continue
        
        has_normal = not sub_df[sub_df['Condition'] == 'NORMAL'].empty
        has_max = not sub_df[sub_df['Condition'] == 'MAX'].empty
        
        if has_normal and has_max:
            normal_vals = sub_df[sub_df['Condition'] == 'NORMAL'][target_cols].mean()
            max_vals = sub_df[sub_df['Condition'] == 'MAX'][target_cols].mean()
            
            ratios = {}
            for col in target_cols:
                n_val = normal_vals[col]
                m_val = max_vals[col]
                if n_val != 0:
                    ratios[col] = m_val / n_val
                else:
                    ratios[col] = np.nan
            
            ratios['Subject ID'] = sub
            ratios['Group'] = group
            ratio_results.append(ratios)
            
    return pd.DataFrame(ratio_results)

# ==========================================
# 4. Statistical Summary Logic
# ==========================================
def calculate_group_statistics(df, target_cols, group_col='Group'):
    """
    各群の Mean ± SD と、Studentに対する各群のp-value (Welch's t-test) を計算して DataFrame で返す。
    ※ ANOVA は除外しました。
    """
    summary_data = []
    groups = ['Student', 'NOCIPN', 'CIPN']
    
    for col in target_cols:
        if col not in df.columns: continue
        
        vals_student = df[df[group_col] == 'Student'][col].dropna()
        vals_nocipn = df[df[group_col] == 'NOCIPN'][col].dropna()
        vals_cipn = df[df[group_col] == 'CIPN'][col].dropna()
        
        stats_res = {'Metric': col}
        
        # Mean ± SD の計算
        for g_name, vals in zip(groups, [vals_student, vals_nocipn, vals_cipn]):
            if len(vals) > 0:
                stats_res[g_name] = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"
            else:
                stats_res[g_name] = "-"

        # Student vs NOCIPN (Welch's t-test)
        if len(vals_student) > 1 and len(vals_nocipn) > 1:
            try:
                _, p_val_nocipn = stats.ttest_ind(vals_student, vals_nocipn, equal_var=False)
                stats_res['p-value (vs NOCIPN)'] = p_val_nocipn
            except:
                stats_res['p-value (vs NOCIPN)'] = np.nan
        else:
            stats_res['p-value (vs NOCIPN)'] = np.nan

        # Student vs CIPN (Welch's t-test)
        if len(vals_student) > 1 and len(vals_cipn) > 1:
            try:
                _, p_val_cipn = stats.ttest_ind(vals_student, vals_cipn, equal_var=False)
                stats_res['p-value (vs CIPN)'] = p_val_cipn
            except:
                stats_res['p-value (vs CIPN)'] = np.nan
        else:
            stats_res['p-value (vs CIPN)'] = np.nan
            
        summary_data.append(stats_res)
        
    return pd.DataFrame(summary_data)

# ==========================================
# 5. Drawing Boxplots & Scatter Logic
# ==========================================
def draw_overlapping_boxplots(df, plot_items, condition_title, output_filename):
    if df.empty: return
    
    num_metrics = len(plot_items)
    fig, axes = plt.subplots(1, num_metrics, figsize=(10 * num_metrics + 2, 10))
    if num_metrics == 1: axes = [axes]
    
    for i, col in enumerate(plot_items):
        ax = axes[i]
        
        vals_student = df[df['Group'] == 'Student'][col].dropna()
        vals_cipn    = df[df['Group'] == 'CIPN'][col].dropna()
        vals_nocipn  = df[df['Group'] == 'NOCIPN'][col].dropna()

        # --- 箱ひげ図 1: Control群 (Student) ---
        if len(vals_student) > 0:
            sns.boxplot(y=vals_student, width=0.5, ax=ax, 
                        boxprops=dict(facecolor="#BDFFA3", edgecolor="#888888", alpha=0.6, linewidth=2),
                        showfliers=False, zorder=1)
        
        # --- 箱ひげ図 2: CIPN群 ---
        if len(vals_cipn) > 0:
            sns.boxplot(y=vals_cipn, width=0.25, ax=ax,
                        boxprops=dict(facecolor="#FFCCCC", edgecolor="red", alpha=0.7, linewidth=3),
                        showfliers=False, zorder=2)

        # --- 散布図プロット（個別有意差の計算付き） ---
        np.random.seed(42)
        control_array = vals_student.values
        
        def plot_scatter_with_sig(vals, dot_color, is_control=False):
            if len(vals) == 0: return
            
            for val in vals:
                is_sig = False
                if not is_control and len(control_array) > 1:
                    n = len(control_array)
                    c_mean = np.mean(control_array)
                    c_std = np.std(control_array, ddof=1)
                    if c_std > 0:
                        t_val = (val - c_mean) / (c_std * np.sqrt((n + 1) / n))
                        p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=n-1))
                        is_sig = p_val < 0.05
                
                if is_sig:
                    marker, size, edge_c, lw, z_ord = ('D', 1000, 'black', 2.5, 20)
                else:
                    marker, size, edge_c, lw, z_ord = ('o', 800, 'white', 2.0, 10)
                
                x_pos = np.random.uniform(-0.06, 0.06)
                ax.scatter(x_pos, val, color=dot_color, marker=marker, s=size, zorder=z_ord,
                           edgecolors=edge_c, linewidth=lw, alpha=0.9)
        
        plot_scatter_with_sig(vals_nocipn, 'blue', is_control=False)    
        plot_scatter_with_sig(vals_cipn, 'red', is_control=False)       

        # --- 軸設定 ---
        ax.set_title(col, fontsize=35, fontweight="bold", pad=20)
        ylabel_text = col if "Ratio" not in condition_title else f"Ratio ({col})"
        ax.set_ylabel(ylabel_text, fontsize=30)
        ax.set_xlabel("Box: Green=Control, Red=CIPN", color="#333333", fontsize=20)
        ax.set_xticks([])
        sns.despine(ax=ax, left=False, bottom=True, top=True, right=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # --- 凡例設定 ---
    custom_lines = [
        Line2D([0], [0], color="#BDFFA3", lw=10, label='Control Box'),
        Line2D([0], [0], color='#FFCCCC', lw=10, label='CIPN Box'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='w', markersize=20, label='CIPN'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markeredgecolor='w', markersize=20, label='Non-CIPN'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='white', markersize=20, label='Non-Sig'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=20, label='Sig (p<0.05)')
    ]

    fig.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(1.0, 0.5), 
               title="Legend", frameon=True, fancybox=True, borderpad=1, fontsize=24, title_fontsize=26)
    
    plt.suptitle(condition_title, fontsize=40, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(right=0.95)
    
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================
# 6. Main Processing
# ==========================================
def main_process_center_4m():
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
        
    search_pattern = os.path.join(INPUT_ROOT_DIR, "**", "*4MWALK*.csv")
    all_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(all_files)} walk files.")
    results = []
    
    for i, fpath in enumerate(all_files):
        fname = os.path.basename(fpath)
        print(f"[{i+1}/{len(all_files)}] Processing {fname} ...", end="", flush=True)
        
        group, subj_id, cond = extract_file_info(fname)

        if group == "CIPN" and subj_id in ["P007", "P008"]:
            print(" Excluded (CIPN P007/P008).")
            continue
        
        metrics = analyze_center_4m_walk(fpath)
        if metrics:
            metrics.update({'Group': group, 'Subject ID': subj_id, 'Condition': cond, 'Filename': fname})
            results.append(metrics)
            print(" Done.")
        else:
            print(" Skipped.")
            
    if not results: 
        print("有効なデータがありませんでした．")
        return

    # 全ファイルの計算結果
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_BASE_DIR, 'walk_center4m_metrics.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")
    
    plot_items = [
        'Gait Speed (m/s)', 'Stride Length (cm)', 'Step Width (cm)', 'Cadence (steps/min)'
    ]

    # Ratioの計算と保存
    df_ratios = calculate_max_normal_ratios(df_res, plot_items)
    if not df_ratios.empty:
        ratio_csv_path = os.path.join(OUTPUT_BASE_DIR, 'walk_center4m_ratios_MaxOverNormal.csv')
        df_ratios.to_csv(ratio_csv_path, index=False)
        print(f"Ratios saved: {ratio_csv_path}")

    # ========================================================
    # 群ごとの平均値・標準偏差と群間検定の結果を 1つの CSV に出力
    # ========================================================
    df_mean = df_res.groupby(['Group', 'Subject ID', 'Condition'])[plot_items].mean().reset_index()

    summary_list = []
    
    # NORMAL / MAX の集計
    for cond in ['NORMAL', 'MAX']:
        df_cond = df_mean[df_mean['Condition'] == cond]
        if df_cond.empty: continue
        
        df_stats = calculate_group_statistics(df_cond, plot_items, group_col='Group')
        df_stats.insert(0, 'Condition', cond)
        summary_list.append(df_stats)
    
    # Ratio の集計
    if not df_ratios.empty:
        df_ratio_stats = calculate_group_statistics(df_ratios, plot_items, group_col='Group')
        df_ratio_stats.insert(0, 'Condition', 'Ratio (MAX/NORMAL)')
        summary_list.append(df_ratio_stats)

    # 結合して出力
    if summary_list:
        df_summary_all = pd.concat(summary_list, ignore_index=True)
        summary_csv_path = os.path.join(OUTPUT_BASE_DIR, 'walk_center4m_group_summary.csv')
        df_summary_all.to_csv(summary_csv_path, index=False)
        print(f"Group Summary with Stats saved: {summary_csv_path}")

    # ========================================================
    # 箱ひげ図の描画
    # ========================================================
    sns.set_style("whitegrid")
    for cond in ['NORMAL', 'MAX']:
        df_plot = df_mean[df_mean['Condition'] == cond]
        if df_plot.empty: continue
        
        out_plot = os.path.join(OUTPUT_BASE_DIR, f'boxplot_metrics_{cond}.png')
        draw_overlapping_boxplots(df_plot, plot_items, f"Walk Metrics ({cond})", out_plot)

    if not df_ratios.empty:
        out_plot_ratio = os.path.join(OUTPUT_BASE_DIR, f'boxplot_ratios_MaxOverNormal.png')
        draw_overlapping_boxplots(df_ratios, plot_items, "MAX / NORMAL Ratios", out_plot_ratio)

    print("\nProcessing completed.")

if __name__ == "__main__":
    main_process_center_4m()