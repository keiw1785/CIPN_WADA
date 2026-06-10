import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os

# ==========================================
# 1. 保存先設定
# ==========================================
output_dir = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results/20260428"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving to: {output_dir}")

# ==========================================
# 2. データ定義
# ==========================================
data = {
    'ID': ['P001', 'P002', 'Non-CIPN', 'P003'], 
    'Group': ['CIPN', 'CIPN', 'Non-CIPN', 'CIPN'], 
    
    # X軸
    'Platinum_Dose': [1690, 2320, 0, 1500],
    'NTX_Total': [33, 20, 0, 13],
    
    # Y軸
    'RR_Path': [1.49, 1.50, 1.16, 1.45],
    'RR_Area': [2.86, 3.58, 1.01, 2.12],
    'RR_AP':   [2.94, 2.46, 0.31, 1.13],
    'RR_ML':   [1.35, 4.86, 1.69, 1.51]
}
df = pd.DataFrame(data)

# プロット用変数
y_vars = ['RR_Path', 'RR_Area', 'RR_AP', 'RR_ML']
y_labels = [
    'Trajectory Length (Path)', 
    'Sway Area (Convex Hull)', 
    'Anterior-Posterior (AP) Sway', 
    'Medial-Lateral (ML) Sway'
]

# デザイン設定
markers = {'Non-CIPN': 'D', 'CIPN': 'o'}
palette = {'Non-CIPN': '#ff7f0e', 'CIPN': '#1f77b4'}

# ==========================================
# 3. スライド用 高視認性プロット関数 (有意差赤字対応)
# ==========================================
def create_slide_svg_red_sig(x_var, x_label, main_title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # 全体のフォント設定
    plt.rcParams.update({'font.size': 18, 'axes.linewidth': 2})

    for i, y_var in enumerate(y_vars):
        ax = axes[i]
        
        # 1. 散布図 (サイズ特大)
        sns.scatterplot(data=df, x=x_var, y=y_var, hue='Group', style='Group',
                        palette=palette, markers=markers, s=500, 
                        edgecolor='black', linewidth=2, ax=ax, legend=False)
        
        # 2. 回帰直線 (太線)
        sns.regplot(data=df, x=x_var, y=y_var, scatter=False, ax=ax, ci=None,
                    line_kws={'color': '#d62728', 'linestyle': '--', 'linewidth': 4})
        
        # 3. 統計量 (条件分岐: p<0.05なら赤字)
        if df[x_var].std() > 0:
            r, p = stats.pearsonr(df[x_var], df[y_var])
            sig_mark = "*" if p < 0.05 else ""
            stats_text = f"R = {r:.2f}\np = {p:.3f}{sig_mark}"
            
            # --- ここが変更点 ---
            # 有意なら赤(#d62728)、そうでなければ黒(#333333)
            if p < 0.05:
                text_color = '#d62728'  # 赤
                edge_color = '#d62728'  # 枠線も赤
                box_lw = 3              # 枠線を太く強調
                bg_alpha = 0.95         # 背景をより白く
            else:
                text_color = '#333333'  # 黒(ダークグレー)
                edge_color = 'gray'     # 枠線はグレー
                box_lw = 2              # 通常の太さ
                bg_alpha = 0.9

        else:
            stats_text = "N/A"
            text_color = '#333333'
            edge_color = 'gray'
            box_lw = 2
            bg_alpha = 0.9
            
        # テキストボックス配置
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, ha='right', va='bottom', 
                fontsize=24, fontweight='bold', color=text_color,
                bbox=dict(facecolor='white', alpha=bg_alpha, edgecolor=edge_color, boxstyle='round,pad=0.4', linewidth=box_lw))

        # 軸ラベルとタイトルの設定
        ax.set_title(y_labels[i], fontsize=24, fontweight='bold', pad=15)
        ax.set_xlabel(x_label, fontsize=20, fontweight='bold')
        ax.set_ylabel('Romberg Ratio', fontsize=20, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=8)
        ax.grid(True, linestyle=':', alpha=0.7, linewidth=1.5)

    # 全体レイアウト
    plt.suptitle(main_title, fontsize=28, fontweight='bold', y=0.98)
    
    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff7f0e', label='Non-CIPN', markersize=18, markeredgecolor='black', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='CIPN', markersize=18, markeredgecolor='black', markeredgewidth=2)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=20, bbox_to_anchor=(0.5, 0.93), frameon=True, edgecolor='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90]) 
    
    # 保存
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================
# 4. 実行
# ==========================================

# 1. 投与量 (Red Sig)
create_slide_svg_red_sig(
    x_var='Platinum_Dose', 
    x_label='Cumulative Platinum Dose (mg)', 
    main_title='Dose-Dependency of Objective Sway', 
    filename='Slide_Dose_RedSig.svg'
)

# 2. NTX (Red Sig)
create_slide_svg_red_sig(
    x_var='NTX_Total', 
    x_label='NTX Total Score (Subjective)', 
    main_title='Correlation: NTX vs Objective Sway', 
    filename='Slide_NTX_RedSig.svg'
)