import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt

# ==========================================
# 設定
# ==========================================
OUTPUT_BASE_DIR = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\daily_results\20260528\trajectory_3m_x_only"
INPUT_ROOT_DIR = r'C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\3D_Result' 

# 閾値設定
TARGET_DISTANCE_X = 3.0  # X方向の距離閾値

# ==========================================
# 共通関数
# ==========================================
def extract_file_info(filename):
    fname_upper = filename.upper()
    if "NOCIPN" in fname_upper:
        group = "NOCIPN"
    elif "CIPN" in fname_upper:
        group = "CIPN"
    elif "STUDENT" in fname_upper:
        group = "STUDENT"
    else:
        group = "Unknown"

    match = re.search(r'(P\d+)', fname_upper)
    subject_id = match.group(1) if match else "Unknown"
    condition = "MAX" if "MAX" in fname_upper else "NORMAL" if "NORMAL" in fname_upper else "Unknown"
    return group, subject_id, condition

def lowpass_filter(data, fps, cutoff=6.0, order=4):
    nyq = 0.5 * fps
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ==========================================
# メイン解析ロジック
# ==========================================
def analyze_tug_trajectory_fixed(file_path, save_dir, filename_base):
    try:
        df = pd.read_csv(file_path).dropna()
        
        # --- 時間軸生成 ---
        if 'TIME' in df.columns:
            time = df['TIME'].values / 1000.0
        else:
            time = np.arange(len(df)) / 30.0
        time = time - time[0]
        dt = np.median(np.diff(time))
        fps = 1.0 / dt if dt > 0 else 30.0

        # --- 座標データの取得 ---
        # 1. 腰 (Hip Center) -> 軌跡描画 & 高さ判定用
        hc_x_raw = (df['left_hip_X'].values + df['right_hip_X'].values) / 2
        hc_z_raw = (df['left_hip_Z'].values + df['right_hip_Z'].values) / 2
        hc_y_raw = (df['left_hip_Y'].values + df['right_hip_Y'].values) / 2

        # Y軸（高さ）の向き補正
        if np.mean(hc_y_raw[len(time)//2-10:len(time)//2+10]) < np.mean(hc_y_raw[:10]):
            y_sign = -1 
        else: 
            y_sign = 1

        # フィルタリング
        hc_x = lowpass_filter(hc_x_raw, fps)
        hc_z = lowpass_filter(hc_z_raw, fps)
        hc_y = lowpass_filter(hc_y_raw * y_sign, fps)
        hc_y = hc_y - np.min(hc_y) # 床面0合わせ

        # 速度計算 (起立着座判定用)
        vx = np.gradient(hc_x, time)
        vy = np.gradient(hc_y, time)
        vz = np.gradient(hc_z, time)
        v_xz = np.sqrt(vx**2 + vz**2) # 水平移動速度
        
        # 軌跡描画用 (スタート地点を0に補正)
        start_x = hc_x[0]
        start_z = hc_z[0]
        traj_x = hc_x - start_x
        traj_z = hc_z - start_z

        # スタート地点(0,0)からの水平距離
        dist_from_start = np.sqrt(traj_x**2 + traj_z**2)

        # ----------------------------------------------------
        # フェーズ検出ロジック
        # ----------------------------------------------------
        
        # --- Phase 3: ターン (Turn) ---
        # 【変更】: 腰(Hip Center)のX変位で判定
        
        # 腰のX方向移動距離 (絶対値)
        dist_x_hc = np.abs(traj_x)

        # 3mを超えているフレームを抽出
        # 救済措置: 腰が3mラインに届かない場合は、最大到達距離の98%を閾値にする
        max_reach_x = np.max(dist_x_hc)
        threshold_x = TARGET_DISTANCE_X
        if max_reach_x < TARGET_DISTANCE_X:
            threshold_x = max_reach_x * 0.98 
        
        in_turn_zone = dist_x_hc > threshold_x
        turn_indices = np.where(in_turn_zone)[0]

        turn_center_idx = np.argmax(dist_from_start) # 補足用: 一番遠い点

        if len(turn_indices) > 0:
            turn_start_idx = turn_indices[0]
            turn_end_idx = turn_indices[-1]
            
            # ノイズ対策: 一瞬だけ超えた場合などを考慮
            if turn_end_idx <= turn_start_idx:
                turn_start_idx = max(0, turn_center_idx - int(1.0*fps))
                turn_end_idx = min(len(time)-1, turn_center_idx + int(1.0*fps))
        else:
            # 3mに誰も到達しなかった場合
            turn_start_idx = max(0, turn_center_idx - int(1.0*fps))
            turn_end_idx = min(len(time)-1, turn_center_idx + int(1.0*fps))


        # --- Phase 1: 起立 (Sit-to-Stand) ---
        # 変更なし: Turnより前で判定
        
        stand_height = np.percentile(hc_y[:turn_start_idx], 85)
        sist_search_lim = int(turn_start_idx * 0.8)
        if sist_search_lim < 1: sist_search_lim = turn_start_idx # ガード
        
        rising_points = np.where((vy[:sist_search_lim] > 0.05) & (hc_y[:sist_search_lim] < stand_height * 0.8))[0]
        
        if len(rising_points) > 0:
            sist_start_idx = rising_points[0]
            for k in range(sist_start_idx, 0, -1):
                if vy[k] <= 0.01:
                    sist_start_idx = k
                    break
        else:
            sist_start_idx = 0
            
        sist_end_candidates = np.where(
            (time > time[sist_start_idx]) & 
            (time < time[turn_start_idx]) & 
            (hc_y > stand_height * 0.90) &
            (vy < 0.05)
        )[0]
        sist_end_idx = sist_end_candidates[0] if len(sist_end_candidates) > 0 else sist_start_idx + int(1.5 * fps)


        # --- Phase 5: 着座 (Sit) ---
        # 【変更】:
        # Start: ターン後、半径0.5m以内に戻った点
        # End: Start以降で「沈み込みピーク」を過ぎて静止した点
        
        # 1. Start判定
        after_turn_indices = np.where((time > time[turn_end_idx]) & (dist_from_start < 1.0))[0]
        
        if len(after_turn_indices) > 0:
            stsi_start_idx = after_turn_indices[0]
        else:
            # 戻りきれなかった場合の救済: 最接近点
            remaining_indices = np.arange(turn_end_idx, len(time))
            if len(remaining_indices) > 0:
                stsi_start_idx = turn_end_idx + np.argmin(dist_from_start[turn_end_idx:])
            else:
                stsi_start_idx = turn_end_idx

        # 2. End判定
        stsi_end_idx = len(time) - 1 # デフォルト末尾
        
        if stsi_start_idx < len(time) - 10:
            # Start以降の vy (垂直速度) を取得
            local_vy = vy[stsi_start_idx:]
            
            # 最大の沈み込み（最小の負の速度）を探す
            min_vy_val = np.min(local_vy)
            min_vy_idx = stsi_start_idx + np.argmin(local_vy)
            
            # 沈み込み動作があったか？ (閾値 -0.1 m/s)
            IS_SINKING_DETECTED = min_vy_val < -0.1

            # 探索開始位置: 沈み込みピークの後から。沈み込みがなければStart直後から
            search_end_start_idx = min_vy_idx if IS_SINKING_DETECTED else stsi_start_idx
            
            # 静止判定閾値
            VEL_XZ_THRES = 0.1   # 水平
            VEL_Y_THRES = 0.05   # 垂直

            for k in range(search_end_start_idx, len(time)):
                # 条件: 水平移動が少なく、かつ 垂直移動もほぼない
                if v_xz[k] < VEL_XZ_THRES and np.abs(vy[k]) < VEL_Y_THRES:
                    # チャタリング防止: この先数フレームも静止しているか
                    if k + 5 < len(time):
                        if np.all(np.abs(vy[k:k+5]) < VEL_Y_THRES):
                            stsi_end_idx = k
                            break
                    else:
                        stsi_end_idx = k
                        break

        # インデックス順序の整合性チェック
        if stsi_end_idx <= stsi_start_idx: stsi_end_idx = len(time) - 1


        # --- Phase 2 & 4: 歩行 ---
        walk_out_start_idx = sist_end_idx
        walk_out_end_idx = turn_start_idx
        walk_back_start_idx = turn_end_idx
        walk_back_end_idx = stsi_start_idx
        
        # 整合性補正 (もし重なりがあれば前のフェーズを優先して縮める)
        if walk_out_start_idx > walk_out_end_idx: walk_out_start_idx = walk_out_end_idx
        if walk_back_start_idx > walk_back_end_idx: walk_back_start_idx = walk_back_end_idx

        # フェーズ格納
        phases = {
            '1_Rise': (sist_start_idx, sist_end_idx),
            '2_WalkOut': (walk_out_start_idx, walk_out_end_idx),
            '3_Turn': (turn_start_idx, turn_end_idx),
            '4_WalkBack': (walk_back_start_idx, walk_back_end_idx),
            '5_Sit': (stsi_start_idx, stsi_end_idx)
        }

        # 秒数計算
        durations = {}
        for k, (s, e) in phases.items():
            durations[k] = max(0, time[e] - time[s])
            
        metrics = {}
        metrics['TUG Score'] = time[stsi_end_idx] - time[sist_start_idx]
        metrics.update(durations)


        # ----------------------------------------------------
        # 可視化: Trajectory Map (X-Z平面)
        # ----------------------------------------------------
        plt.figure(figsize=(8, 10))
        
        colors = {'1_Rise': 'red', '2_WalkOut': 'orange', '3_Turn': 'purple', '4_WalkBack': 'green', '5_Sit': 'blue'}
        
        # 3mライン描画
        direction = 1 if np.mean(traj_x[turn_center_idx-10:turn_center_idx+10]) > 0 else -1
        target_line_x = TARGET_DISTANCE_X * direction
        plt.axvline(target_line_x, color='gray', linestyle='--', linewidth=2, label='3m Line (Hip)')

        # 軌跡プロット
        for key, (s, e) in phases.items():
            if s < e:
                plt.plot(traj_x[s:e], traj_z[s:e], color=colors[key], linewidth=3, label=key)
                
                # 秒数テキスト
                mid_idx = (s + e) // 2
                plt.text(traj_x[mid_idx], traj_z[mid_idx], f"{durations[key]:.2f}s", 
                         color=colors[key], fontsize=11, fontweight='bold', 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        # スタート地点
        plt.scatter(0, 0, color='black', marker='x', s=100, label='Start')

        plt.title(f"TUG Analysis (Hip X Check): {filename_base}\nTotal: {metrics['TUG Score']:.2f} sec", fontsize=14)
        plt.xlabel("Progression X (m)")
        plt.ylabel("Lateral Z (m)")
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"Trajectory_Fixed_{filename_base}.png")
        plt.savefig(save_path)
        plt.close()

        return metrics

    except Exception as e:
        print(f"Error {filename_base}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# メイン実行
# ==========================================
def main():
    if not os.path.exists(OUTPUT_BASE_DIR): os.makedirs(OUTPUT_BASE_DIR)
    files = glob.glob(os.path.join(INPUT_ROOT_DIR, "**", "*TUG*.csv"), recursive=True)
    
    print(f"Processing {len(files)} files...")
    results = []
    
    for f in files:
        fname = os.path.basename(f)
        grp, subj, cond = extract_file_info(fname)

        save_dir = os.path.join(OUTPUT_BASE_DIR, grp, subj)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        res = analyze_tug_trajectory_fixed(f, save_dir, os.path.splitext(fname)[0])
        if res:
            res.update({'File': fname, 'Group': grp, 'Subject': subj, 'Condition': cond})
            results.append(res)
            
    if results:
        df_res = pd.DataFrame(results)
        # カラム順序を整理
        cols = ['Group', 'Subject', 'Condition', 'TUG Score', '1_Rise', '2_WalkOut', '3_Turn', '4_WalkBack', '5_Sit', 'File']
        df_res = df_res[cols]
        
        df_res.to_csv(os.path.join(OUTPUT_BASE_DIR, 'summary_trajectory_fixed.csv'), index=False)
        print("Done.")

if __name__ == "__main__":
    main()