import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === パス設定 ===
base_path = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA"
input_base = os.path.join(base_path, "data", "1_processed", "main_research")
output_base = os.path.join(base_path, r"data\2_time_series_feature\main_research\CoG")

# ▼▼▼ フィルタリング設定 ▼▼▼
# 被験者グループの指定（例: "NOCIPN"）
# ※ 全グループを処理する場合は "" (空文字) にしてください
target_group = "CIPN" 

# 被験者IDの指定（例: "P002"）
# ※ 全被験者を処理する場合は "" (空文字) にしてください
target_subject_id = "P005" 
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# === スケール設定 ===
SCALE_MM_PER_PX = 1680 / 1080  # ≒ 1.556 mm/px

# === 日本語フォント設定 ===
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# === ViTPoseの関節定義（共通） ===
JOINTS = {
    "nose": {"x": "nose_X", "y": "nose_Y"},
    "left_eye": {"x": "left_eye_X", "y": "left_eye_Y"},
    "right_eye": {"x": "right_eye_X", "y": "right_eye_Y"},
    "left_ear": {"x": "left_ear_X", "y": "left_ear_Y"},
    "right_ear": {"x": "right_ear_X", "y": "right_ear_Y"},
    "left_shoulder": {"x": "left_shoulder_X", "y": "left_shoulder_Y"},
    "right_shoulder": {"x": "right_shoulder_X", "y": "right_shoulder_Y"},
    "left_elbow": {"x": "left_elbow_X", "y": "left_elbow_Y"},
    "right_elbow": {"x": "right_elbow_X", "y": "right_elbow_Y"},
    "left_wrist": {"x": "left_wrist_X", "y": "left_wrist_Y"},
    "right_wrist": {"x": "right_wrist_X", "y": "right_wrist_Y"},
    "left_hip": {"x": "left_hip_X", "y": "left_hip_Y"},
    "right_hip": {"x": "right_hip_X", "y": "right_hip_Y"},
    "left_knee": {"x": "left_knee_X", "y": "left_knee_Y"},
    "right_knee": {"x": "right_knee_X", "y": "right_knee_Y"},
    "left_ankle": {"x": "left_ankle_X", "y": "left_ankle_Y"},
    "right_ankle": {"x": "right_ankle_X", "y": "right_ankle_Y"},
}

# === 1. 質量比 ===
# 全身モデル用 (C2)
BODY_SEGMENTS_MASS_RATIO_FULL = {
    "head": 0.081, "trunk": 0.497,
    "right_upper_arm": 0.028, "left_upper_arm": 0.028,
    "right_forearm": 0.016, "left_forearm": 0.016,
    "right_hand": 0.006, "left_hand": 0.006,
    "right_thigh": 0.100, "left_thigh": 0.100,
    "right_shin": 0.0465, "left_shin": 0.0465,
    "right_foot": 0.0145, "left_foot": 0.0145,
}

# 左半身モデル用 (C1)
BODY_SEGMENTS_MASS_RATIO_LEFT = {
    "head": 0.081 / 2, "trunk": 0.497 / 2,
    "left_upper_arm": 0.028, "left_forearm": 0.016,
    "left_hand": 0.006, "left_thigh": 0.100,
    "left_shin": 0.0465, "left_foot": 0.0145,
}

# === 2. 重心位置の比率 (Proximal Ratio) ===
COM_PROXIMAL_RATIOS = {
    "upper_arm": 0.436, "forearm": 0.430,
    "thigh": 0.433, "shin": 0.433,
}

# === 3. セグメントの構造定義 ===
# 全身モデル用 (C2)
SEGMENT_PAIRS_FULL = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shin": ("left_knee", "left_ankle"),
    "right_shin": ("right_knee", "right_ankle"),
}
SEGMENT_OTHERS_FULL = {
    "head": ("nose", "left_ear", "right_ear"),
    "trunk": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "left_hand": ("left_wrist",), "right_hand": ("right_wrist",),
    "left_foot": ("left_ankle",), "right_foot": ("right_ankle",),
}

# 左半身モデル用 (C1)
SEGMENT_PAIRS_LEFT = {
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "left_shin": ("left_knee", "left_ankle"),
}
SEGMENT_OTHERS_LEFT = {
    "head": ("left_ear",), "trunk": ("left_shoulder", "left_hip"),
    "left_hand": ("left_wrist",), "left_foot": ("left_ankle",),
}

# === 処理開始 ===
# グループ指定の有無で検索階層を調整
group_query = target_group if target_group else "**"
search_pattern = os.path.join(input_base, group_query, "**", "ROMBERG", "**", "*.csv")
all_csv_files = glob.glob(search_pattern, recursive=True)

# IDによるフィルタリング
if target_subject_id:
    csv_files = [f for f in all_csv_files if target_subject_id in f]
    print(f"フィルタ設定: グループ '{target_group or 'すべて'}', ID '{target_subject_id}' のROMBERGファイルのみ処理します．")
else:
    csv_files = all_csv_files
    print(f"フィルタ設定: グループ '{target_group or 'すべて'}', 全IDのROMBERGファイルを処理します．")

if not csv_files:
    print("対象のCSVファイルが見つかりませんでした．")
else:
    print(f"{len(csv_files)} 件のCSVファイルを検出しました．")

for csv_path in csv_files:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # 出力先ディレクトリの構築（input_base からの相対パスを使用することで、グループ/ID階層を自動維持）
    rel_dir = os.path.relpath(os.path.dirname(csv_path), input_base)
    out_subdir = os.path.join(output_base, rel_dir)
    os.makedirs(out_subdir, exist_ok=True)

    print(f"\n▶ {rel_dir}\\{base_name}.csv を処理中...")

    # C1 か C2 かの判定をパスから実施
    path_parts = os.path.normpath(csv_path).split(os.sep)
    is_c1 = "C1" in path_parts
    is_c2 = "C2" in path_parts

    if not is_c1 and not is_c2:
        print("  ⚠ パスに C1 または C2 が含まれていないためスキップします．")
        continue

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠ 読み込みエラー: {e}")
        continue

    # 判定結果に基づいてモデルのパラメータをセット
    if is_c1:
        req_joints = ["left_shoulder_X", "left_hip_X", "left_knee_X", "left_ankle_X"]
        seg_pairs = SEGMENT_PAIRS_LEFT
        seg_others = SEGMENT_OTHERS_LEFT
        mass_ratio = BODY_SEGMENTS_MASS_RATIO_LEFT
        model_name = "左半身モデル/C1"
    else:
        req_joints = ["left_shoulder_X", "right_shoulder_X", "left_hip_X", "right_hip_X"]
        seg_pairs = SEGMENT_PAIRS_FULL
        seg_others = SEGMENT_OTHERS_FULL
        mass_ratio = BODY_SEGMENTS_MASS_RATIO_FULL
        model_name = "全身モデル/C2"

    if not all(col in df.columns for col in req_joints):
        print(f"  ⚠ 必須関節カラムが不足しています（{model_name}）．スキップ．")
        continue

    # 欠損・0値フレーム除去
    df = df.replace(0, np.nan).dropna(subset=req_joints)
    if df.empty:
        print("  ⚠ 有効なフレームがありません．")
        continue

    # --- 重心計算 ---
    df["CoG_X"] = 0.0
    df["CoG_Y"] = 0.0
    total_mass = sum(mass_ratio.values())

    # 1. 四肢
    for segment, (prox_name, dist_name) in seg_pairs.items():
        if JOINTS[prox_name]["x"] not in df.columns or JOINTS[dist_name]["x"] not in df.columns:
            continue
        
        ratio_key = segment.replace("left_", "").replace("right_", "")
        ratio = COM_PROXIMAL_RATIOS.get(ratio_key, 0.5)

        prox_x = df[JOINTS[prox_name]["x"]]
        prox_y = df[JOINTS[prox_name]["y"]]
        dist_x = df[JOINTS[dist_name]["x"]]
        dist_y = df[JOINTS[dist_name]["y"]]

        cx = prox_x + (dist_x - prox_x) * ratio
        cy = prox_y + (dist_y - prox_y) * ratio

        m = mass_ratio[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m

    # 2. その他
    for segment, landmarks in seg_others.items():
        valid = [name for name in landmarks if JOINTS[name]["x"] in df.columns]
        if not valid:
            continue

        if segment == "trunk" and not is_c1: # C2（全身）の体幹計算
            s_cx = (df["left_shoulder_X"] + df["right_shoulder_X"]) / 2
            h_cx = (df["left_hip_X"] + df["right_hip_X"]) / 2
            cx = (s_cx + h_cx) / 2
            s_cy = (df["left_shoulder_Y"] + df["right_shoulder_Y"]) / 2
            h_cy = (df["left_hip_Y"] + df["right_hip_Y"]) / 2
            cy = (s_cy + h_cy) / 2
        else: # C1の体幹または他のセグメント
            cx = sum(df[JOINTS[n]["x"]] for n in valid) / len(valid)
            cy = sum(df[JOINTS[n]["y"]] for n in valid) / len(valid)
        
        m = mass_ratio[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m

    # 総質量で割る（全身比率の合計が1.0にならないケースや半身モデルの場合への対応）
    df["CoG_X"] /= total_mass
    df["CoG_Y"] /= total_mass

    # --- スケール変換（px → mm） ---
    df["CoG_X_mm"] = df["CoG_X"] * SCALE_MM_PER_PX
    df["CoG_Y_mm"] = df["CoG_Y"] * SCALE_MM_PER_PX

    # --- 出力CSV ---
    out_csv = os.path.join(out_subdir, f"{base_name}_CoG.csv")
    df[["TIME", "CoG_X_mm", "CoG_Y_mm"]].to_csv(out_csv, index=False)
    print(f"  ✅ 出力: {out_csv}")

    # --- プロット ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"重心推移 ({model_name}) - {base_name}", fontsize=15)
    t = df["TIME"]

    axes[0].plot(t, df["CoG_X_mm"], color="r", label="X軸 [mm]")
    axes[0].set_ylabel("X [mm]")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, df["CoG_Y_mm"], color="g", label="Y軸 [mm]")
    axes[1].set_ylabel("Y [mm]")
    axes[1].set_xlabel("時間 [ms]")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")
    
    # C1（側面視など）の場合はY軸を反転させて見やすくする
    if is_c1:
        axes[1].invert_yaxis()

    out_plot = os.path.join(out_subdir, f"{base_name}_CoG.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_plot)
    plt.close(fig)
    print(f"  📈 グラフ保存: {out_plot}")

print("\n=== 全ファイルの処理が完了しました ===")