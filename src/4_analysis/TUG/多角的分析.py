import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# ★ 設定エリア ★
# ==========================================
# 指定された入力CSVファイルのパス
INPUT_CSV_PATH = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results\20260528/TUG/tug_final_ratio_metric/final_svg/tug_metrics_averaged.csv"

# 出力先フォルダ (入力ファイルと同じ場所に 'advanced_analysis_results' を作成)
INPUT_DIR = os.path.dirname(INPUT_CSV_PATH)
OUTPUT_BASE_DIR = os.path.join(INPUT_DIR, "advanced_analysis_results")

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif']
sns.set(font=['MS Gothic', 'Meiryo', 'Yu Gothic', 'sans-serif'])

# 解析対象の指標リスト
TARGET_METRICS = [
    'TUG Score (s)', 
    'Total Steps', 
    'Turn Radius (m)', 
    'Turn Duration (s)', 
    'Stride Length Ratio', 
    'Rise Duration (s)',
    'Sit Duration (s)'
]

# ==========================================
# 1. 可視化・解析関数群 (前回と同じ)
# ==========================================
def plot_radar_chart_zscore(df_cond, target_cols, condition_name, output_dir):
    """ レーダーチャート """
    student_data = df_cond[df_cond['Group'] == 'Student']
    if len(student_data) == 0: return

    mean_std = student_data[target_cols].mean()
    std_std = student_data[target_cols].std()
    
    df_z = df_cond.copy()
    for col in target_cols:
        if std_std[col] == 0: df_z[col] = 0
        else: df_z[col] = (df_cond[col] - mean_std[col]) / std_std[col]

    df_z_grouped = df_z.groupby('Group')[target_cols].mean()
    
    categories = target_cols; N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]; angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], categories, color='black', size=12)
    plt.ylim(-3, 3)
    plt.yticks([-2, -1, 0, 1, 2], ["-2σ", "-1σ", "Avg", "+1σ", "+2σ"], color="grey", size=10)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    colors = {'Student': 'gray', 'NOCIPN': 'blue', 'CIPN': 'red'}
    for grp, color in colors.items():
        if grp in df_z_grouped.index:
            val = df_z_grouped.loc[grp].values.flatten().tolist(); val += val[:1]
            width = 3 if grp == 'CIPN' else 1
            style = 'solid' if grp != 'Student' else '--'
            ax.plot(angles, val, linewidth=width, linestyle=style, label=grp, color=color)
            if grp != 'Student': ax.fill(angles, val, color, alpha=0.1)

    plt.title(f"Radar Chart (Z-score) - {condition_name}", size=18, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(os.path.join(output_dir, f"Radar_{condition_name}.svg"), format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"Radar_{condition_name}.png"), format='png', bbox_inches='tight')  
    plt.close()

def plot_heatmap(df_cond, target_cols, condition_name, output_dir):
    """ ヒートマップ """
    df_hm = df_cond[['Subject', 'Group'] + target_cols].copy().set_index('Subject')
    for col in target_cols:
        m, s = df_hm[col].mean(), df_hm[col].std()
        df_hm[col] = (df_hm[col] - m) / s if s != 0 else 0

    lut = {'Student': 'skyblue', 'NOCIPN': 'blue', 'CIPN': 'red'}
    row_colors = df_hm['Group'].map(lut)
    g = sns.clustermap(df_hm[target_cols], row_colors=row_colors, cmap="vlag", center=0, col_cluster=False, figsize=(10, 8))
    g.fig.suptitle(f"Heatmap - {condition_name}", y=1.02, fontsize=16)
    plt.savefig(os.path.join(output_dir, f"Heatmap_{condition_name}.svg"), format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"Heatmap_{condition_name}.png"), format='png', bbox_inches='tight')    
    plt.close()

def plot_effect_sizes(df_cond, target_cols, condition_name, output_dir):
    """ 効果量プロット """
    stu = df_cond[df_cond['Group'] == 'Student']
    cipn = df_cond[df_cond['Group'] == 'CIPN']
    if len(stu) < 2 or len(cipn) < 2: return

    effect_sizes = []
    for col in target_cols:
        m1, m2 = stu[col].mean(), cipn[col].mean()
        s1, s2 = stu[col].std(), cipn[col].std()
        n1, n2 = len(stu), len(cipn)
        pooled_sd = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
        d = (m2 - m1) / pooled_sd if pooled_sd != 0 else 0
        effect_sizes.append({'Metric': col, 'Effect Size (d)': d})
        
    df_es = pd.DataFrame(effect_sizes).sort_values(by='Effect Size (d)', key=abs, ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_es, x='Effect Size (d)', y='Metric', palette='coolwarm')
    plt.axvline(0.8, color='gray', linestyle='--'); plt.axvline(-0.8, color='gray', linestyle='--')
    plt.title(f"Effect Size (Cohen's d) - {condition_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"EffectSize_{condition_name}.svg"), format='svg')
    plt.savefig(os.path.join(output_dir, f"EffectSize_{condition_name}.png"), format='png')   
    plt.close()

def plot_pca_scatter(df_cond, target_cols, condition_name, output_dir):
    """ PCA散布図 & 定量評価 """
    if len(df_cond) < 3: return
    X = df_cond[target_cols].fillna(df_cond[target_cols].mean())
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2); X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca['Group'] = df_cond['Group'].values
    df_pca['Subject'] = df_cond['Subject'].values

    pc1_s = df_pca[df_pca['Group'] == 'Student']['PC1']
    pc1_c = df_pca[df_pca['Group'] == 'CIPN']['PC1']
    stats_text = ""
    if len(pc1_s) > 1 and len(pc1_c) > 1:
        t, p = stats.ttest_ind(pc1_s, pc1_c, equal_var=False)
        stats_text = f"PC1 T-test: p={p:.3f}" + ("**" if p < 0.01 else "*" if p < 0.05 else "")

    plt.figure(figsize=(10, 8))
    from scipy.spatial import ConvexHull
    pts = df_pca[df_pca['Group'] == 'Student'][['PC1', 'PC2']].values
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        plt.fill(pts[hull.vertices,0], pts[hull.vertices,1], 'skyblue', alpha=0.1)
        plt.plot(pts[hull.vertices,0], pts[hull.vertices,1], 'k--', alpha=0.3)

    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Group', style='Group', 
                    palette={'Student': 'skyblue', 'NOCIPN': 'blue', 'CIPN': 'red'}, s=200)
    
    for i in range(len(df_pca)):
        if df_pca.Group[i] in ['CIPN', 'NOCIPN']:
            plt.text(df_pca.PC1[i]+0.2, df_pca.PC2[i], df_pca.Subject[i], color='darkred', weight='bold')

    plt.title(f"PCA Map ({condition_name})\n{stats_text}", fontsize=16)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"); plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.axhline(0, color='gray', linestyle=':'); plt.axvline(0, color='gray', linestyle=':')
    plt.savefig(os.path.join(output_dir, f"PCA_{condition_name}.svg"), format='svg')
    plt.savefig(os.path.join(output_dir, f"PCA_{condition_name}.png"), format='png')   
    plt.close()

def plot_composite_score(df_cond, target_cols, condition_name, output_dir):
    """ 総合異常度スコア """
    stu_df = df_cond[df_cond['Group'] == 'Student']
    if len(stu_df) == 0: return
    
    mean_vec = stu_df[target_cols].mean(); std_vec = stu_df[target_cols].std()
    z_scores = pd.DataFrame()
    for col in target_cols:
        z_scores[col] = ((df_cond[col] - mean_vec[col]) / std_vec[col]).abs() if std_vec[col] != 0 else 0
    
    df_res = df_cond.copy()
    df_res['Composite_Score'] = z_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_res, x='Group', y='Composite_Score', order=['Student', 'NOCIPN', 'CIPN'],
                palette={'Student': 'lightskyblue', 'NOCIPN': 'blue', 'CIPN': 'red'}, showfliers=False)
    sns.stripplot(data=df_res, x='Group', y='Composite_Score', order=['Student', 'NOCIPN', 'CIPN'],
                  color='black', size=8, alpha=0.7)

    s_vals = df_res[df_res['Group']=='Student']['Composite_Score']
    c_vals = df_res[df_res['Group']=='CIPN']['Composite_Score']
    if len(s_vals) > 1 and len(c_vals) > 1:
        t, p = stats.ttest_ind(s_vals, c_vals, equal_var=False)
        sig_mark = f"p={p:.3f}" + ("**" if p < 0.01 else "*" if p < 0.05 else "")
        y_max = df_res['Composite_Score'].max()
        plt.plot([0, 2], [y_max*1.05, y_max*1.05], 'k-', lw=1.5)
        plt.text(1, y_max*1.08, sig_mark, ha='center', va='bottom', fontsize=14, color='red')

    plt.title(f"Composite Score ({condition_name})", fontsize=16)
    plt.savefig(os.path.join(output_dir, f"CompositeScore_{condition_name}.svg"), format='svg')
    plt.savefig(os.path.join(output_dir, f"CompositeScore_{condition_name}.png"), format='png')    
    plt.close()

# ==========================================
# 2. メイン実行フロー
# ==========================================
def main():
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    
    # 1. データの読み込み
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return
    
    print(f"Loading data from: {INPUT_CSV_PATH}")
    df_mean = pd.read_csv(INPUT_CSV_PATH)

    # 2. Ratioデータの生成 (CSVに既にあればそれを使い、なければ計算する)
    if 'RATIO' in df_mean['Condition'].unique():
        print("Ratio data found in CSV. Using it directly.")
        df_all = df_mean
    else:
        print("Calculating Ratio (MAX/NORMAL)...")
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
            print("Warning: Failed to calculate Ratio. MAX or NORMAL data might be missing.")
            df_all = df_mean

    # 3. 全条件ループ解析
    for cond in ['NORMAL', 'MAX', 'RATIO']:
        print(f"\n--- Analysis: {cond} ---")
        df_cond = df_all[df_all['Condition'] == cond]
        if len(df_cond) == 0: 
            print("No data found.")
            continue
        
        plot_radar_chart_zscore(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        plot_heatmap(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        plot_effect_sizes(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        plot_pca_scatter(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)
        plot_composite_score(df_cond, TARGET_METRICS, cond, OUTPUT_BASE_DIR)

    print(f"\nCompleted! Check output folder: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()