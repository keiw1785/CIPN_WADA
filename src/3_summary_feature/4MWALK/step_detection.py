import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats

# ==========================================
# Settings: Output and Input Paths
# ==========================================
# Output folder
OUTPUT_BASE_DIR = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results\20260528\4M\walk_analysis_normalized_ratio_with_stats"

# Input data root folder
INPUT_ROOT_DIR = r'C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\3D_Result' 

# フォント設定（文字化け防止・Windows用）
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
def analyze_center_4m_walk(file_path, save_dir, filename_base):
    try:
        df = pd.read_csv(file_path).dropna()
        
        # Time Axis
        if 'TIME' in df.columns:
            time = df['TIME'].values / 1000.0
        else:
            time = np.arange(len(df)) / 30.0
        time = time - time[0]
        dt = np.median(np.diff(time))
        fps = 1.0 / dt if dt > 0 else 30.0

        # --- Axis Auto-Detection ---
        cols = ['left_hip', 'right_hip', 'left_ankle', 'right_ankle']
        axes = ['X', 'Y', 'Z']
        
        # Forward axis (Largest range)
        ranges = {}
        for ax in axes:
            vals = df[f'left_hip_{ax}'].values
            ranges[ax] = np.max(vals) - np.min(vals)
        fwd_axis = max(ranges, key=ranges.get)
        
        # Vertical axis
        h_s_diffs = {}
        for ax in axes:
            if ax == fwd_axis: continue
            h = (df[f'left_hip_{ax}'] + df[f'right_hip_{ax}']) / 2
            s = (df[f'left_shoulder_{ax}'] + df[f'right_shoulder_{ax}']) / 2
            h_s_diffs[ax] = np.mean(np.abs(s - h))
        vert_axis = max(h_s_diffs, key=h_s_diffs.get)
        
        # Lateral axis
        lat_axis = [ax for ax in axes if ax not in [fwd_axis, vert_axis]][0]

        # --- Trimming: Extract Center 4m ---
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
        
        # --- Parameter Calculation ---
        l_fwd = lowpass_filter(df_ss[f'left_ankle_{fwd_axis}'].values, fps)
        r_fwd = lowpass_filter(df_ss[f'right_ankle_{fwd_axis}'].values, fps)
        l_lat = lowpass_filter(df_ss[f'left_ankle_{lat_axis}'].values, fps)
        r_lat = lowpass_filter(df_ss[f'right_ankle_{lat_axis}'].values, fps)
        
        l_wrist_fwd = lowpass_filter(df_ss[f'left_wrist_{fwd_axis}'].values, fps)
        l_hip_fwd = lowpass_filter(df_ss[f'left_hip_{fwd_axis}'].values, fps)
        r_wrist_fwd = lowpass_filter(df_ss[f'right_wrist_{fwd_axis}'].values, fps)
        r_hip_fwd = lowpass_filter(df_ss[f'right_hip_{fwd_axis}'].values, fps)

        # 1. Calculate Leg Length (for Normalization)
        l_leg_dist = np.sqrt(
            (df_ss[f'left_hip_{fwd_axis}'] - df_ss[f'left_ankle_{fwd_axis}'])**2 +
            (df_ss[f'left_hip_{lat_axis}'] - df_ss[f'left_ankle_{lat_axis}'])**2 +
            (df_ss[f'left_hip_{vert_axis}'] - df_ss[f'left_ankle_{vert_axis}'])**2
        )
        r_leg_dist = np.sqrt(
            (df_ss[f'right_hip_{fwd_axis}'] - df_ss[f'right_ankle_{fwd_axis}'])**2 +
            (df_ss[f'right_hip_{lat_axis}'] - df_ss[f'right_ankle_{lat_axis}'])**2 +
            (df_ss[f'right_hip_{vert_axis}'] - df_ss[f'right_ankle_{vert_axis}'])**2
        )
        avg_leg_length_m = (np.mean(l_leg_dist) + np.mean(r_leg_dist)) / 2.0
        avg_leg_length_cm = avg_leg_length_m * 100

        # 2. Gait Speed
        analyzed_dist = np.abs(ank_fwd[mask][-1] - ank_fwd[mask][0])
        duration = time_ss[-1]
        speed = analyzed_dist / duration if duration > 0 else 0
        
        # 3. Step Detection
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
            sl = np.abs(l_fwd[p] - r_fwd[p]) * 100 # cm
            stride_lengths.append(sl)
            sw = np.abs(l_lat[p] - r_lat[p]) * 100 # cm
            step_widths.append(sw)
            
        # --- Calculate Mean, Variance, CV, Normalized ---
        if len(stride_lengths) > 0:
            avg_stride = np.mean(stride_lengths)
            var_stride = np.var(stride_lengths, ddof=1) if len(stride_lengths) > 1 else 0
            cv_stride = (np.std(stride_lengths, ddof=1) / avg_stride * 100) if (len(stride_lengths) > 1 and avg_stride > 0) else 0
            norm_stride = avg_stride / avg_leg_length_cm if avg_leg_length_cm > 0 else 0
        else:
            avg_stride = 0; var_stride = 0; cv_stride = 0; norm_stride = 0

        if len(step_widths) > 0:
            avg_width = np.mean(step_widths)
            var_width = np.var(step_widths, ddof=1) if len(step_widths) > 1 else 0
            cv_width = (np.std(step_widths, ddof=1) / avg_width * 100) if (len(step_widths) > 1 and avg_width > 0) else 0
        else:
            avg_width = 0; var_width = 0; cv_width = 0
        
        step_count = len(peaks)
        cadence = (step_count / duration) * 60 if duration > 0 else 0
        
        l_swing = np.max(l_wrist_fwd - l_hip_fwd) - np.min(l_wrist_fwd - l_hip_fwd)
        r_swing = np.max(r_wrist_fwd - r_hip_fwd) - np.min(r_wrist_fwd - r_hip_fwd)
        avg_arm_swing = (l_swing + r_swing) / 2 * 100 # cm
        
        metrics = {
            "Leg Length (cm)": round(avg_leg_length_cm, 2),
            "Gait Speed (m/s)": round(speed, 2),
            "Stride Length (cm)": round(avg_stride, 2),
            "Norm Stride Length": round(norm_stride, 3),
            "Stride Length Var": round(var_stride, 2),
            "Stride Length CV (%)": round(cv_stride, 2),
            "Step Width (cm)": round(avg_width, 2),
            "Step Width Var": round(var_width, 2),
            "Step Width CV (%)": round(cv_width, 2),
            "Cadence (steps/min)": round(cadence, 1),
            "Arm Swing (cm)": round(avg_arm_swing, 2),
            "Step Count": step_count
        }
        
        # Save Plot (Individual Track)
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        ax[0].plot(l_fwd, l_lat, label='Left'); ax[0].plot(r_fwd, r_lat, label='Right')
        ax[0].set_title(f'Path ({filename_base})'); ax[0].legend(); ax[0].axis('equal')
        ax[1].plot(time_ss, dist_ankles); ax[1].plot(time_ss[peaks], dist_ankles[peaks], "x")
        ax[1].set_title('Step Detection')
        ax[2].plot(time_ss, (l_wrist_fwd - l_hip_fwd)*100); ax[2].plot(time_ss, (r_wrist_fwd - r_hip_fwd)*100)
        ax[2].set_title('Arm Swing')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"WalkCenter4m_{filename_base}.png"))
        plt.close()
        
        return metrics

    except Exception as e:
        print(f"Error in {filename_base}: {e}")
        return None

# ==========================================
# 3. Ratio Calculation Logic
# ==========================================
def calculate_max_normal_ratios(df_results):
    ratio_results = []
    
    target_cols = [
        'Gait Speed (m/s)', 'Stride Length (cm)', 'Norm Stride Length', 
        'Stride Length Var', 'Stride Length CV (%)', 'Step Width (cm)', 
        'Step Width Var', 'Step Width CV (%)', 'Cadence (steps/min)', 'Arm Swing (cm)'
    ]
    
    subjects = df_results['Subject ID'].unique()
    
    for sub in subjects:
        sub_df = df_results[df_results['Subject ID'] == sub]
        if sub_df.empty: continue
        group = sub_df['Group'].iloc[0]
        
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
    summary_data = []
    groups = ['Student', 'NOCIPN', 'CIPN']
    
    for col in target_cols:
        if col not in df.columns: continue
        
        vals_student = df[df[group_col] == 'Student'][col].dropna()
        vals_nocipn = df[df[group_col] == 'NOCIPN'][col].dropna()
        vals_cipn = df[df[group_col] == 'CIPN'][col].dropna()
        
        stats_res = {'Metric': col}
        
        # Mean ± SD
        for g_name, vals in zip(groups, [vals_student, vals_nocipn, vals_cipn]):
            if len(vals) > 0:
                stats_res[g_name] = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"
            else:
                stats_res[g_name] = "-"
        
        # ANOVA
        if len(vals_student) > 1 and len(vals_nocipn) > 1 and len(vals_cipn) > 1:
            try:
                _, p_val = stats.f_oneway(vals_student, vals_nocipn, vals_cipn)
                stats_res['p-value'] = p_val
            except:
                stats_res['p-value'] = np.nan
        else:
            stats_res['p-value'] = np.nan
            
        summary_data.append(stats_res)
        
    return pd.DataFrame(summary_data)

def save_statistical_table(df_stats, title, output_path):
    if df_stats.empty: return

    col_labels = ['Metric', 'Student', 'NOCIPN', 'CIPN', 'p-value']
    cell_text = []
    
    for i in range(len(df_stats)):
        row_data = df_stats.iloc[i]
        
        p = row_data['p-value']
        if pd.isna(p): p_str = "-"
        elif p < 0.001: p_str = "< 0.001"
        else: p_str = f"{p:.3f}"
        
        cell_text.append([row_data['Metric'], row_data['Student'], row_data['NOCIPN'], row_data['CIPN'], p_str])

    plt.figure(figsize=(14, len(df_stats) * 0.6 + 1.5))
    ax = plt.gca()
    ax.axis('off')
    
    table = plt.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', colWidths=[0.3, 0.2, 0.2, 0.2, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    for i in range(len(df_stats)):
        p_val = df_stats.iloc[i]['p-value']
        is_sig = (not pd.isna(p_val)) and (p_val < 0.05)
        
        bg_color = '#ffe6e6' if is_sig else 'white'
        text_weight = 'bold' if is_sig else 'normal'

        table_row_idx = i + 1
        for j in range(len(col_labels)):
            cell = table[table_row_idx, j]
            cell.set_facecolor(bg_color)
            cell.set_text_props(weight=text_weight)
            if j == 4 and is_sig:
                cell.set_text_props(color='red', weight='bold')

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')

    plt.title(title, fontsize=16, pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Table saved: {output_path}")

# ==========================================
# 5. Drawing Boxplots & Scatter Logic (Overlapping Style)
# ==========================================
def draw_overlapping_boxplots(df, plot_items, condition_title, output_filename):
    if df.empty: return
    
    num_metrics = len(plot_items)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 10))
    if num_metrics == 1: axes = [axes]
    
    for i, col in enumerate(plot_items):
        if col not in df.columns: continue
        ax = axes[i]
        
        vals_student = df[df['Group'] == 'Student'][col].dropna()
        vals_nocipn  = df[df['Group'] == 'NOCIPN'][col].dropna()
        vals_cipn    = df[df['Group'] == 'CIPN'][col].dropna()

        # --- 箱ひげ図の描画 (Studentのみ) ---
        if len(vals_student) > 0:
            sns.boxplot(y=vals_student, width=0.6, ax=ax, 
                        boxprops=dict(facecolor="#BDFFA3", edgecolor="#00b050", alpha=0.6, linewidth=2),
                        showfliers=False, zorder=1)

        # --- 散布図プロット ---
        np.random.seed(42)
        def plot_scatter(vals, dot_color, zorder):
            if len(vals) == 0: return
            jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(jitter, vals, color=dot_color, marker='o', s=800, 
                       edgecolors='white', linewidth=2.0, alpha=0.9, zorder=zorder)
        
        plot_scatter(vals_student, '#00b050', zorder=10) # Student: 緑
        plot_scatter(vals_nocipn, '#0070c0', zorder=11)  # NOCIPN: 青
        plot_scatter(vals_cipn, '#ff0000', zorder=12)    # CIPN: 赤
        
        # --- 軸の装飾 ---
        ax.set_title(col, fontsize=24, fontweight="bold", pad=20)
        ax.set_ylabel(col if "Ratio" not in condition_title else f"Ratio ({col})", fontsize=20)
        ax.set_xlabel("Groups", fontsize=18)
        ax.set_xticks([]) # X軸のラベル（目盛り）は消去
        sns.despine(ax=ax, left=False, bottom=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # --- カスタム凡例の追加 ---
    custom_lines = [
        Line2D([0], [0], color="#BDFFA3", lw=10, label='Student Box'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00b050', markersize=20, label='Student'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0070c0', markersize=20, label='NOCIPN'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff0000', markersize=20, label='CIPN')
    ]
    # 右端に凡例を配置するため、余白を調整
    fig.legend(handles=custom_lines, loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=20, title="Legend", title_fontsize=22)
    
    plt.suptitle(condition_title, fontsize=30, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(right=0.95) # 凡例用に右側に余白を空ける
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
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
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        group, subj_id, cond = extract_file_info(fname)
        subj_dir = os.path.join(OUTPUT_BASE_DIR, group, subj_id)
        if not os.path.exists(subj_dir): os.makedirs(subj_dir)
        
        metrics = analyze_center_4m_walk(fpath, subj_dir, os.path.splitext(fname)[0])
        if metrics:
            metrics.update({'Group': group, 'Subject ID': subj_id, 'Condition': cond, 'Filename': fname})
            results.append(metrics)
            
    if not results: return

    # データの保存
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_BASE_DIR, 'walk_center4m_metrics.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")
    
    df_ratios = calculate_max_normal_ratios(df_res)
    if not df_ratios.empty:
        ratio_csv_path = os.path.join(OUTPUT_BASE_DIR, 'walk_center4m_ratios_MaxOverNormal.csv')
        df_ratios.to_csv(ratio_csv_path, index=False)
        print(f"Ratios saved: {ratio_csv_path}")
    
    # 描画対象の指標
    plot_items = [
        'Gait Speed (m/s)', 'Norm Stride Length', 'Stride Length CV (%)',
        'Step Width (cm)', 'Step Width CV (%)', 'Cadence (steps/min)', 'Arm Swing (cm)'
    ]

    sns.set_style("whitegrid")
    df_mean = df_res.groupby(['Group', 'Subject ID', 'Condition'])[plot_items].mean().reset_index()

    # --- A. Analyze Normal & Max Conditions ---
    for cond in ['NORMAL', 'MAX']:
        df_plot = df_mean[df_mean['Condition'] == cond]
        
        # オーバーラップ箱ひげ図の出力
        out_plot = os.path.join(OUTPUT_BASE_DIR, f'boxplot_metrics_{cond}.png')
        draw_overlapping_boxplots(df_plot, plot_items, f"Walk Metrics ({cond}) - Subject Averages", out_plot)

        # 統計表の出力
        df_stats = calculate_group_statistics(df_plot, plot_items, group_col='Group')
        save_statistical_table(df_stats, f"Statistical Summary ({cond})", os.path.join(OUTPUT_BASE_DIR, f"Stats_Table_{cond}.png"))

    # --- B. Analyze Ratios ---
    if not df_ratios.empty:
        # オーバーラップ箱ひげ図の出力 (Ratio)
        out_plot_ratio = os.path.join(OUTPUT_BASE_DIR, f'boxplot_ratios_MaxOverNormal.png')
        draw_overlapping_boxplots(df_ratios, plot_items, "MAX / NORMAL Ratios", out_plot_ratio)

        # 統計表の出力
        df_stats_ratio = calculate_group_statistics(df_ratios, plot_items, group_col='Group')
        save_statistical_table(df_stats_ratio, "Statistical Summary (Ratio: MAX/NORMAL)", os.path.join(OUTPUT_BASE_DIR, "Stats_Table_Ratio.png"))

    print("\nProcessing completed.")

if __name__ == "__main__":
    main_process_center_4m()