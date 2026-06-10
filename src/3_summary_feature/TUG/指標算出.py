import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
src_root = Path(__file__).resolve().parents[2]
sys.path.append(str(src_root))
from utils.config import get_project_root
# ==========================================
# ★ スライド用・視認性強化設定 ★
# ==========================================
# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif']

# スタイル定数 (ここを変えると全体のサイズが一括で変わります)
FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS_LABEL = 30
FONT_SIZE_TICK = 32
FONT_SIZE_LEGEND = 32
FONT_SIZE_TEXT = 20       # プロット内の文字など
FONT_SIZE_P_MARK = 30     # 星マーク (**)
MARKER_SIZE = 800         # 散布図の点の大きさ
LINE_WIDTH_BOX = 3.5      # 箱ひげの線の太さ
LINE_WIDTH_GRID = 3.0     # グリッド線の太さ

# ==========================================
# 設定 (Settings)
# ==========================================
BASE_PATH = get_project_root()
OUTPUT_BASE_DIR = BASE_PATH / "daily_results" / "20260528" / "TUG" / "tug_final_ratio_metric" / "final_svg"
INPUT_ROOT_DIR = BASE_PATH / "data" / "1_processed" / "3D_Result"
AVG_CSV_PATH = OUTPUT_BASE_DIR / "tug_metrics_averaged.csv"
EXCLUDE_SUBJECTS = ["P007",'P008'] #STUDENTのうち、除くデータ

# 解析用パラメータ
TARGET_DISTANCE_X = 3.0 
STEP_HEIGHT_THRES = 0.05 
STEP_MIN_DIST_SEC = 0.25

# ==========================================
# 統計関数: Crawford's t-test
# ==========================================
def crawford_t_test(control_sample, single_val):
    n = len(control_sample)
    if n < 2: return np.nan, np.nan
    mean_c = np.mean(control_sample)
    sd_c = np.std(control_sample, ddof=1)
    t_stat = (single_val - mean_c) / (sd_c * np.sqrt((n + 1) / n))
    df = n - 1
    p_val = stats.t.sf(np.abs(t_stat), df) * 2
    return t_stat, p_val

# ==========================================
# 1. 共通関数
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
    nyq = 0.5 * fps; normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def fit_circle(x, y):
    x = np.array(x); y = np.array(y)
    if len(x) < 3: return np.nan
    x_m = np.mean(x); y_m = np.mean(y)
    u = x - x_m; v = y - y_m
    Suu = np.sum(u**2); Svv = np.sum(v**2); Suv = np.sum(u*v)
    Suuu = np.sum(u**3); Svvv = np.sum(v**3)
    Suvv = np.sum(u*(v**2)); Svuu = np.sum(v*(u**2))
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([0.5*(Suuu + Suvv), 0.5*(Svvv + Svuu)])
    try:
        center_uv = np.linalg.solve(A, B)
        return np.sqrt(center_uv[0]**2 + center_uv[1]**2 + (Suu + Svv)/len(x))
    except:
        return np.nan

# ==========================================
# 2. 解析ロジック
# ==========================================
def analyze_tug_final(file_path):
    try:
        df = pd.read_csv(file_path).dropna()
        if 'TIME' in df.columns: time = df['TIME'].values / 1000.0
        else: time = np.arange(len(df)) / 30.0
        time = time - time[0]
        dt = np.median(np.diff(time))
        fps = 1.0 / dt if dt > 0 else 30.0

        hc_x_raw = (df['left_hip_X'].values + df['right_hip_X'].values) / 2
        hc_z_raw = (df['left_hip_Z'].values + df['right_hip_Z'].values) / 2
        hc_y_raw = (df['left_hip_Y'].values + df['right_hip_Y'].values) / 2
        sc_x_raw = (df['left_shoulder_X'].values + df['right_shoulder_X'].values) / 2
        sc_z_raw = (df['left_shoulder_Z'].values + df['right_shoulder_Z'].values) / 2
        sc_y_raw = (df['left_shoulder_Y'].values + df['right_shoulder_Y'].values) / 2
        l_ank_x_raw = df['left_ankle_X'].values; l_ank_z_raw = df['left_ankle_Z'].values
        r_ank_x_raw = df['right_ankle_X'].values; r_ank_z_raw = df['right_ankle_Z'].values

        y_sign = -1 if np.mean(sc_y_raw) < np.mean(hc_y_raw) else 1
        def clean(d, s=1): return lowpass_filter(d * s, fps)

        hc_x = clean(hc_x_raw); hc_z = clean(hc_z_raw)
        hc_y = clean(hc_y_raw, y_sign); hc_y -= np.min(hc_y)
        sc_x = clean(sc_x_raw); sc_z = clean(sc_z_raw); sc_y = clean(sc_y_raw, y_sign)
        l_ank_x = clean(l_ank_x_raw); l_ank_z = clean(l_ank_z_raw)
        r_ank_x = clean(r_ank_x_raw); r_ank_z = clean(r_ank_z_raw)

        traj_x = hc_x - hc_x[0]; traj_z = hc_z - hc_z[0]
        dist_from_start = np.sqrt(traj_x**2 + traj_z**2)
        vx = np.gradient(hc_x, time); vy = np.gradient(hc_y, time)
        vz = np.gradient(hc_z, time); v_xz = np.sqrt(vx**2 + vz**2)
        heading_rad = np.arctan2(vz, vx)
        yaw_rate_deg_s = np.abs(np.degrees(np.gradient(np.unwrap(heading_rad), time)))

        vec_x = sc_x - hc_x; vec_y = sc_y - hc_y; vec_z = sc_z - hc_z
        vec_len = np.sqrt(vec_x**2 + vec_y**2 + vec_z**2)
        abs_angle_deg = np.degrees(np.arccos(np.clip(vec_y / vec_len, -1.0, 1.0)))
        baseline_angle = np.mean(abs_angle_deg[:int(0.5*fps)])
        trunk_flexion_deg = np.abs(abs_angle_deg - baseline_angle)

        dist_x_hc = np.abs(traj_x)
        max_reach_x = np.max(dist_x_hc)
        threshold_x = TARGET_DISTANCE_X if max_reach_x >= TARGET_DISTANCE_X else max_reach_x * 0.98
        in_turn_zone = dist_x_hc > threshold_x
        turn_indices = np.where(in_turn_zone)[0]
        turn_center_idx = np.argmax(dist_from_start)

        if len(turn_indices) > 0:
            turn_s = turn_indices[0]; turn_e = turn_indices[-1]
            if turn_e <= turn_s:
                turn_s = max(0, turn_center_idx - int(1.0*fps))
                turn_e = min(len(time)-1, turn_center_idx + int(1.0*fps))
        else:
            turn_s = max(0, turn_center_idx - int(1.0*fps))
            turn_e = min(len(time)-1, turn_center_idx + int(1.0*fps))

        stand_height = np.percentile(hc_y[:turn_s], 90)
        sist_search_lim = max(1, int(turn_s * 0.8))
        rising = np.where((vy[:sist_search_lim] > 0.05) & (hc_y[:sist_search_lim] < stand_height * 0.8))[0]
        sist_s = rising[0] if len(rising) > 0 else 0
        for k in range(sist_s, 0, -1):
            if vy[k] <= 0.01: sist_s = k; break
        sist_e_cands = np.where((time > time[sist_s]) & (time < time[turn_s]) & (hc_y > stand_height * 0.90) & (vy < 0.05))[0]
        sist_e = sist_e_cands[0] if len(sist_e_cands) > 0 else sist_s + int(1.5 * fps)

        after_turn = np.where((time > time[turn_e]) & (dist_from_start < 1.0))[0]
        stsi_s = after_turn[0] if len(after_turn) > 0 else turn_e
        stsi_e = len(time) - 1
        if stsi_s < len(time) - 10:
            local_vy = vy[stsi_s:]
            search_start = stsi_s + np.argmin(local_vy) if np.min(local_vy) < -0.1 else stsi_s
            for k in range(search_start, len(time)):
                if v_xz[k] < 0.1 and np.abs(vy[k]) < 0.05:
                    if k+5 < len(time) and np.all(np.abs(vy[k:k+5]) < 0.05):
                        stsi_e = k; break
                    else: stsi_e = k; break
        if stsi_e <= stsi_s: stsi_e = len(time)-1

        metrics = {}
        metrics['TUG Score (s)'] = time[stsi_e] - time[sist_s]
        
        t_x = traj_x[turn_s:turn_e]; t_z = traj_z[turn_s:turn_e]
        metrics['Turn Radius (m)'] = fit_circle(t_x, t_z)
        
        # ★★★ 変更点: 平均角速度を削除し、旋回時間に変更 ★★★
        # turn_ang_vel = yaw_rate_deg_s[turn_s:turn_e]
        # metrics['Turn AngVel Mean (deg/s)'] = np.mean(turn_ang_vel) if len(turn_ang_vel)>0 else 0
        metrics['Turn Duration (s)'] = time[turn_e] - time[turn_s]
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★
        
        ankle_dist = np.sqrt((l_ank_x - r_ank_x)**2 + (l_ank_z - r_ank_z)**2)
        walk_s = sist_e; walk_e = stsi_s
        if walk_e > walk_s:
            pks_all, _ = find_peaks(ankle_dist[walk_s:walk_e], height=STEP_HEIGHT_THRES, distance=int(fps*STEP_MIN_DIST_SEC))
            pks_all = pks_all + walk_s 
            metrics['Total Steps'] = len(pks_all)
        else: metrics['Total Steps'] = 0
        
        valid_strides = []
        if len(pks_all) > 1:
            for i in range(len(pks_all)-1):
                idx1 = pks_all[i]; idx2 = pks_all[i+1]
                step_mid = (idx1 + idx2) // 2
                is_turn_step = (step_mid >= turn_s) and (step_mid <= turn_e)
                if not is_turn_step:
                    dist = np.sqrt((hc_x[idx2]-hc_x[idx1])**2 + (hc_z[idx2]-hc_z[idx1])**2)
                    valid_strides.append(dist)
        
        if len(valid_strides) > 1:
            max_sl = np.max(valid_strides); min_sl = np.min(valid_strides)
            if min_sl > 0.01: metrics['Stride Length Ratio'] = max_sl / min_sl
            else: metrics['Stride Length Ratio'] = np.nan
        else: metrics['Stride Length Ratio'] = np.nan

        metrics['Rise Duration (s)'] = time[sist_e] - time[sist_s]
        metrics['Sit Duration (s)'] = time[stsi_e] - time[stsi_s]
        metrics['Rise Max Flexion (deg)'] = np.max(trunk_flexion_deg[sist_s:sist_e]) if sist_e > sist_s else 0
        metrics['Sit Max Flexion (deg)'] = np.max(trunk_flexion_deg[stsi_s:stsi_e]) if stsi_e > stsi_s else 0

        return metrics
    except Exception as e: return None

# ==========================================
# 3. 描画・統計関数 (視認性強化版)
# ==========================================
def plot_at_position(ax, df_subset, item_name, position, label_prefix=""):
    stats_res = {}
    student_vals = df_subset[df_subset['Group'] == 'Student'][item_name].dropna().values
    nocipn_vals = df_subset[df_subset['Group'] == 'NOCIPN'][item_name].dropna().values
    cipn_subset = df_subset[df_subset['Group'] == 'CIPN'][['Subject', item_name]].dropna()
    cipn_vals = cipn_subset[item_name].values
    cipn_subjs = cipn_subset['Subject'].values

    # Healthy Boxplot (太く、濃く)
    if len(student_vals) > 0:
        ax.boxplot(student_vals, positions=[position], patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor='lightskyblue', color='black', alpha=0.9, linewidth=LINE_WIDTH_BOX),
                   medianprops=dict(color='black', linewidth=LINE_WIDTH_BOX),
                   whiskerprops=dict(color='black', linewidth=LINE_WIDTH_BOX),
                   capprops=dict(color='black', linewidth=LINE_WIDTH_BOX), showfliers=False)
    
    jitter_width = 0.06
    # NOCIPN Scatter (大きく)
    if len(nocipn_vals) > 0:
        x_nocipn = np.random.normal(position, jitter_width, size=len(nocipn_vals))
        ax.scatter(x_nocipn, nocipn_vals, color='blue', s=MARKER_SIZE, alpha=0.9, edgecolors='white', linewidth=2, zorder=10)
    
    # CIPN Scatter (大きく、ラベルも大きく)
    if len(cipn_vals) > 0:
        x_cipn = np.random.normal(position, jitter_width, size=len(cipn_vals))
        ax.scatter(x_cipn, cipn_vals, color='red', s=MARKER_SIZE, alpha=0.9, edgecolors='white', linewidth=2, zorder=10)
        for x, y, subj in zip(x_cipn, cipn_vals, cipn_subjs):
            ax.text(x + 0.03, y, subj, fontsize=FONT_SIZE_TEXT, color='darkred', weight='bold', ha='left', va='center', zorder=12)

    all_vals = np.concatenate([student_vals, nocipn_vals, cipn_vals]) if (len(student_vals)+len(nocipn_vals)+len(cipn_vals)) > 0 else np.array([0])
    y_max = np.max(all_vals)
    y_step = y_max * 0.10 # 間隔を広げる
    curr_y = y_max + y_step * 0.5

    # --- 統計 (p値) ---
    # vs NOCIPN
    if len(student_vals) > 1 and len(nocipn_vals) > 0:
        if len(nocipn_vals) == 1:
            t, p = crawford_t_test(student_vals, nocipn_vals[0])
        else:
            t, p = stats.ttest_ind(student_vals, nocipn_vals, equal_var=False)
        
        stats_res[f'{label_prefix}p_NOCIPN'] = p
        if p < 0.05:
            mark = "**" if p < 0.01 else "*"
            # マークを大きく
            ax.text(position - 0.2, curr_y, mark, ha='center', va='center', color='blue', fontsize=FONT_SIZE_P_MARK, fontweight='bold')
            curr_y += y_step
    else: stats_res[f'{label_prefix}p_NOCIPN'] = np.nan

    # vs CIPN
    if len(student_vals) > 1 and len(cipn_vals) > 0:
        if len(cipn_vals) == 1:
            t, p = crawford_t_test(student_vals, cipn_vals[0])
        else:
            t, p = stats.ttest_ind(student_vals, cipn_vals, equal_var=False)
        
        stats_res[f'{label_prefix}p_CIPN'] = p
        if p < 0.05:
            mark = "**" if p < 0.01 else "*"
            # マークを大きく
            ax.text(position + 0.2, curr_y, mark, ha='center', va='center', color='red', fontsize=FONT_SIZE_P_MARK, fontweight='bold')
    else: stats_res[f'{label_prefix}p_CIPN'] = np.nan

    prefix = label_prefix if label_prefix else ""
    stats_res[f'{prefix}Student_Mean'] = np.mean(student_vals) if len(student_vals)>0 else np.nan
    stats_res[f'{prefix}Student_SD'] = np.std(student_vals, ddof=1) if len(student_vals)>1 else np.nan
    stats_res[f'{prefix}Student_N'] = len(student_vals)
    stats_res[f'{prefix}NOCIPN_Mean'] = np.mean(nocipn_vals) if len(nocipn_vals)>0 else np.nan
    stats_res[f'{prefix}NOCIPN_SD'] = np.std(nocipn_vals, ddof=1) if len(nocipn_vals)>1 else np.nan
    stats_res[f'{prefix}NOCIPN_N'] = len(nocipn_vals)
    stats_res[f'{prefix}CIPN_Mean'] = np.mean(cipn_vals) if len(cipn_vals)>0 else np.nan
    stats_res[f'{prefix}CIPN_SD'] = np.std(cipn_vals, ddof=1) if len(cipn_vals)>1 else np.nan
    stats_res[f'{prefix}CIPN_N'] = len(cipn_vals)

    return stats_res

# ==========================================
# 4. 表画像保存 (SVG対応 & 拡大版)
# ==========================================
def format_and_save_stats_table(df_stats, output_path, cols_prefix="Ratio_"):
    if df_stats.empty: return

    def fmt_mean_sd(mean, sd, count):
        if count <= 0: return "-"
        if count == 1 or pd.isna(sd): return f"{mean:.3f}"
        return f"{mean:.3f} ± {sd:.3f}"

    def fmt_pval(p):
        if pd.isna(p): return "-"
        if p < 0.001: return "< 0.001"
        return f"{p:.3f}"

    formatted_data = []
    for i, row in df_stats.iterrows():
        metric = row.get('Metric', '')
        n_stu = row.get(f'{cols_prefix}Student_N', 0)
        stu_str = fmt_mean_sd(row.get(f'{cols_prefix}Student_Mean', np.nan), row.get(f'{cols_prefix}Student_SD', np.nan), n_stu)
        
        n_no = row.get(f'{cols_prefix}NOCIPN_N', 0)
        no_str = fmt_mean_sd(row.get(f'{cols_prefix}NOCIPN_Mean', np.nan), row.get(f'{cols_prefix}NOCIPN_SD', np.nan), n_no)
        p_no = row.get(f'{cols_prefix}p_NOCIPN', np.nan)
        
        n_cipn = row.get(f'{cols_prefix}CIPN_N', 0)
        cipn_str = fmt_mean_sd(row.get(f'{cols_prefix}CIPN_Mean', np.nan), row.get(f'{cols_prefix}CIPN_SD', np.nan), n_cipn)
        p_cipn = row.get(f'{cols_prefix}p_CIPN', np.nan)

        formatted_data.append({
            'Metric': metric,
            'Healthy (Mean±SD)': stu_str,
            'NOCIPN (Mean±SD)': no_str,
            'p (vs NO)': fmt_pval(p_no),
            'CIPN (Mean±SD)': cipn_str,
            'p (vs CIPN)': fmt_pval(p_cipn),
            '_raw_p_NOCIPN': p_no, '_raw_p_CIPN': p_cipn
        })
    
    df_fmt = pd.DataFrame(formatted_data)
    display_cols = [c for c in df_fmt.columns if not c.startswith('_raw')]
    
    # 図として保存 (SVG)
    # 幅と高さを大きく確保
    fig_height = len(df_fmt) * 1.0 + 2.0
    fig, ax = plt.subplots(figsize=(20, fig_height)) 
    ax.axis('off')
    
    table = ax.table(cellText=df_fmt[display_cols].values, colLabels=display_cols, loc='center', cellLoc='center')
    
    # 文字サイズ拡大
    table.auto_set_font_size(False)
    table.set_fontsize(18) 
    table.scale(1.0, 3.0) # 行の高さ拡大
    
    for (row, col), cell in table.get_celld().items():
        if row == 0: 
            cell.set_facecolor('#e0e0e0')
            cell.set_text_props(weight='bold', fontsize=20)

    col_list = display_cols
    for r_idx in range(len(df_fmt)):
        table_row = r_idx + 1
        raw_p_no = df_fmt.iloc[r_idx]['_raw_p_NOCIPN']
        raw_p_cipn = df_fmt.iloc[r_idx]['_raw_p_CIPN']
        
        if 'p (vs NO)' in col_list:
            c = col_list.index('p (vs NO)')
            if pd.notna(raw_p_no) and raw_p_no < 0.05:
                table[table_row, c].set_facecolor('#ffcccc')
                table[table_row, c].set_text_props(color='red', weight='bold', fontsize=20)
        if 'p (vs CIPN)' in col_list:
            c = col_list.index('p (vs CIPN)')
            if pd.notna(raw_p_cipn) and raw_p_cipn < 0.05:
                table[table_row, c].set_facecolor('#ffcccc')
                table[table_row, c].set_text_props(color='red', weight='bold', fontsize=20)

    # SVG出力
    save_path = output_path.replace('.png', '.svg')
    plt.savefig(save_path, bbox_inches='tight', format='svg')
    plt.close()
    
    # CSVも保存
    csv_path = output_path.replace('.png', '.csv').replace('.svg', '.csv')
    df_fmt[display_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved Stats Table (SVG): {save_path}")

def create_and_save_comparison_plot(plot_items, filename, title_suffix, df_mean, legend_handles):
    n_plots = len(plot_items); cols = 3; rows = (n_plots + cols - 1) // cols
    # フィギュアサイズを大きく
    fig_height = 8 * rows if rows > 0 else 8
    fig, axes = plt.subplots(rows, cols, figsize=(24, fig_height)) # 幅広
    if rows == 1 and cols == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    summary_data = [] 
    for i, item in enumerate(plot_items):
        ax = axes[i]
        df_norm = df_mean[df_mean['Condition'] == 'NORMAL']
        df_max  = df_mean[df_mean['Condition'] == 'MAX']
        row_stats = {'Metric': item}
        
        res_norm = plot_at_position(ax, df_norm, item, 1, label_prefix="NORMAL_")
        row_stats.update(res_norm)
        res_max = plot_at_position(ax, df_max, item, 2, label_prefix="MAX_")
        row_stats.update(res_max)
        
        summary_data.append(row_stats)
        ax.set_title(item, fontsize=FONT_SIZE_TITLE, pad=20)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['NORMAL', 'MAX'], fontsize=FONT_SIZE_AXIS_LABEL)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        ax.set_xlim(0.3, 2.7)
        ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=LINE_WIDTH_GRID)
        
    for j in range(len(plot_items), len(axes)): axes[j].axis('off')
    
    fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=FONT_SIZE_LEGEND)
    plt.suptitle(f'TUG Metrics Comparison: {title_suffix}', fontsize=FONT_SIZE_TITLE+4, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # SVG出力
    save_name = filename.replace('.png', '.svg')
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, save_name), format='svg')
    plt.close()
    print(f"Saved Plot (SVG): {save_name}")
    
    if summary_data:
        df_sum = pd.DataFrame(summary_data)
        # 表出力 (ファイル名の拡張子を処理)
        base_name = filename.replace('.png', '').replace('.svg', '')
        format_and_save_stats_table(df_sum, os.path.join(OUTPUT_BASE_DIR, f'{base_name}_stats_normal.svg'), cols_prefix="NORMAL_")
        format_and_save_stats_table(df_sum, os.path.join(OUTPUT_BASE_DIR, f'{base_name}_stats_max.svg'), cols_prefix="MAX_")

def main():
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    files = glob.glob(os.path.join(INPUT_ROOT_DIR, "**", "*TUG*.csv"), recursive=True)
    results = []
    for fpath in files:
        fname = os.path.basename(fpath)
        group, subj_id, cond = extract_file_info(fname)
        if subj_id in EXCLUDE_SUBJECTS: continue
        res = analyze_tug_final(fpath)
        if res:
            res.update({'Group': group, 'Subject': subj_id, 'Condition': cond})
            results.append(res)
    if not results: return

    df_raw = pd.DataFrame(results)
    df_mean = df_raw.groupby(['Group', 'Subject', 'Condition'], as_index=False).mean(numeric_only=True)
    df_mean.to_csv(os.path.join(OUTPUT_BASE_DIR, 'tug_metrics_averaged.csv'), index=False)

    # 凡例マーカーサイズ調整
    patch_student = mpatches.Patch(color='lightskyblue', label='Healthy')
    patch_nocipn = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=18, label='NOCIPN')
    patch_cipn = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=18, label='CIPN')
    legend_handles = [patch_student, patch_nocipn, patch_cipn]
    
    sns.set_style("whitegrid")

    # 1. 全9指標 (変更点: Turn AngVel Mean -> Turn Duration)
    plot_items_9 = ['TUG Score (s)', 'Total Steps', 'Turn Radius (m)', 'Turn Duration (s)', 'Stride Length Ratio', 'Rise Duration (s)', 'Rise Max Flexion (deg)', 'Sit Duration (s)', 'Sit Max Flexion (deg)']
    create_and_save_comparison_plot(plot_items_9, 'combined_plot_normal_max.svg', 'NORMAL vs MAX', df_mean, legend_handles)

    # 2. 追加要望1
    plot_items_subset_1 = ['TUG Score (s)', 'Total Steps']
    create_and_save_comparison_plot(plot_items_subset_1, 'selected_metrics_time_steps.svg', 'Time & Steps', df_mean, legend_handles)

    # 3. 追加要望2 (変更点: Turn AngVel Mean -> Turn Duration)
    plot_items_subset_2 = ['Rise Duration (s)', 'Turn Duration (s)', 'Turn Radius (m)']
    create_and_save_comparison_plot(plot_items_subset_2, 'selected_metrics_rise_turn.svg', 'Rise & Turn', df_mean, legend_handles)

    # 4. Ratio Plot
    df_pivot = df_mean.pivot(index=['Group', 'Subject'], columns='Condition', values=plot_items_9)
    df_ratio = pd.DataFrame(index=df_pivot.index) 
    for item in plot_items_9:
        if ('MAX' in df_pivot[item].columns) and ('NORMAL' in df_pivot[item].columns):
            df_ratio[item] = df_pivot[item]['MAX'] / df_pivot[item]['NORMAL']
    df_ratio = df_ratio.reset_index()
    
    # サイズ調整
    fig2, axes2 = plt.subplots(3, 3, figsize=(24, 20))
    axes2 = axes2.flatten()
    summary_ratio = []
    for i, item in enumerate(plot_items_9):
        if i >= len(axes2): break
        ax = axes2[i]
        row_stats = {'Metric': item}
        res = plot_at_position(ax, df_ratio, item, 1, label_prefix="Ratio_")
        row_stats.update(res)
        ax.set_title(f"Ratio: {item}", fontsize=FONT_SIZE_TITLE, pad=15)
        ax.set_xticks([]); ax.set_xlabel('Ratio (MAX/NORMAL)', fontsize=FONT_SIZE_AXIS_LABEL)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK)
        ax.set_xlim(0.5, 1.5); ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        summary_ratio.append(row_stats)
    for j in range(len(plot_items_9), len(axes2)): axes2[j].axis('off')
    
    fig2.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=FONT_SIZE_LEGEND)
    plt.suptitle('TUG Metrics Ratio (MAX / NORMAL)', fontsize=FONT_SIZE_TITLE+4, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # SVG出力
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, 'all_metrics_ratio_plot.svg'), format='svg')
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, 'all_metrics_ratio_plot.svg'), format='png')   
    plt.close()
    print("Saved All Ratio Plot (SVG).")
    
    if summary_ratio:
        df_ratio_stat = pd.DataFrame(summary_ratio)
        format_and_save_stats_table(df_ratio_stat, os.path.join(OUTPUT_BASE_DIR, 'ratio_stats_formatted.svg'), cols_prefix="Ratio_")

if __name__ == "__main__":
    main()