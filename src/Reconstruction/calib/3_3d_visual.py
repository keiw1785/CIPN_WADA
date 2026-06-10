import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------
# 1. FFmpegとファイルのパス設定
# ---------------------------------------------------------
# FFmpegのパス
FFMPEG_PATH = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\etc\ffmpeg\bin\ffmpeg.exe"
plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

# CSVファイルのパス
csv_path = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\3D_Result\CIPN\P005\TUG\detected_CIPN-TUG-P005-20260528-NORMAL-T3-3D_trim.csv"
# ---------------------------------------------------------
# 2. データ読み込み & 前処理
# ---------------------------------------------------------
df = pd.read_csv(csv_path)

# 【重要】欠損値（空欄）がある行を削除
df = df.dropna()

if len(df) == 0:
    raise ValueError("データが空です。")

# 関節定義
joints = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

bones_colored = [
    (0, 1, 'gray'), (0, 2, 'gray'), (1, 3, 'gray'), (2, 4, 'gray'),
    (5, 6, 'black'), (5, 11, 'black'), (6, 12, 'black'), (11, 12, 'black'),
    (5, 7, 'blue'), (7, 9, 'blue'), (11, 13, 'cornflowerblue'), (13, 15, 'cornflowerblue'),
    (6, 8, 'red'), (8, 10, 'red'), (12, 14, 'salmon'), (14, 16, 'salmon')
]

# データ整形 & 正規化
T = len(df)
data_norm = np.zeros((T, len(joints), 3))

for i, joint in enumerate(joints):
    data_norm[:, i, 0] = df[f'{joint}_X'].values
    data_norm[:, i, 1] = df[f'{joint}_Z'].values   # Y(奥行)
    data_norm[:, i, 2] = -df[f'{joint}_Y'].values  # Z(高さ)

# 最小値を0にシフト
data_norm[:, :, 0] -= np.nanmin(data_norm[:, :, 0])
data_norm[:, :, 1] -= np.nanmin(data_norm[:, :, 1])
data_norm[:, :, 2] -= np.nanmin(data_norm[:, :, 2])

# 各軸の最大値（レンジ）を取得
max_x = np.nanmax(data_norm[:, :, 0])
max_y = np.nanmax(data_norm[:, :, 1])
max_z = np.nanmax(data_norm[:, :, 2])

# ---------------------------------------------------------
# 3. アニメーション設定（余白削除版）
# ---------------------------------------------------------
# 横長の動きに合わせて横長の図にする
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# 【改善点1】グラフ周囲の余白を削除
plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

# 軸の範囲
ax.set_xlim(0, max_x)
ax.set_ylim(0, max_y)
ax.set_zlim(0, max_z)

# アスペクト比をデータに合わせる（歪み防止）
ax.set_box_aspect((max_x, max_y, max_z))

# 【改善点2】カメラの距離を近づける (デフォルトは10)
# 値を小さくするほどズームします。7〜8くらいが適当です。
ax.dist = 7.5

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('4M Walk Analysis (Zoomed)')
ax.view_init(elev=20, azim=90) #視点変更

# 描画オブジェクト
lines = [ax.plot([], [], [], color=c, lw=2)[0] for _, _, c in bones_colored]
points, = ax.plot([], [], [], 'ko', markersize=3)
trail, = ax.plot([], [], [], 'g--', lw=1, alpha=0.5)

l_hip, r_hip = 11, 12
hip_pos = (data_norm[:, l_hip, :] + data_norm[:, r_hip, :]) / 2

def update(frame):
    if frame >= len(data_norm): return lines + [points, trail]
    current = data_norm[frame]
    
    points.set_data(current[:, 0], current[:, 1])
    points.set_3d_properties(current[:, 2])
    
    for line, (j1, j2, _) in zip(lines, bones_colored):
        line.set_data([current[j1, 0], current[j2, 0]], [current[j1, 1], current[j2, 1]])
        line.set_3d_properties([current[j1, 2], current[j2, 2]])
        
    trail.set_data(hip_pos[:frame+1, 0], hip_pos[:frame+1, 1])
    trail.set_3d_properties(hip_pos[:frame+1, 2])
    
    return lines + [points, trail]

ani = animation.FuncAnimation(fig, update, frames=range(0, T, 2), interval=30, blit=False)

# ---------------------------------------------------------
# 4. 保存
# ---------------------------------------------------------
print("MP4保存を開始します...")
ani.save(r'C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results\20260528\skeleton_walk_zoomed.mp4', writer='ffmpeg', fps=30)
print("保存完了: skeleton_walk_zoomed.mp4")