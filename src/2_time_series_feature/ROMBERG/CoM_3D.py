import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
src_root = Path(__file__).resolve().parents[2]
sys.path.append(str(src_root))
from utils.config import get_project_root

# === パス設定 ===
BASE_PATH = get_project_root()
input_base = BASE_PATH / "data" / "1_processed" / "3D_Result"
output_base = BASE_PATH / "data" / "2_time_series_feature" / "main_research" / "CoG_calib"

# ▼▼▼ フィルタリング設定 ▼▼▼
# 被験者グループの指定（例: "NOCIPN"）
# ※ 全グループを処理する場合は "" (空文字) にしてください
target_group = "" 

# 被験者IDの指定（例: "P002"）
# ※ 全被験者を処理する場合は "" (空文字) にしてください
target_subject_id = "" 
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# === 日本語フォント設定 ===
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['axes.unicode_minus'] = False

# === ViTPoseの関節定義（共通） ===
JOINTS = {
    "nose": {"x": "nose_X", "y": "nose_Y", "z": "nose_Z"},
    "left_eye": {"x": "left_eye_X", "y": "left_eye_Y", "z": "left_eye_Z" },
    "right_eye": {"x": "right_eye_X", "y": "right_eye_Y", "z": "right_eye_Z"},
    "left_ear": {"x": "left_ear_X", "y": "left_ear_Y", "z": "left_ear_Z"},
    "right_ear": {"x": "right_ear_X", "y": "right_ear_Y", "z": "right_ear_Z"},
    "left_shoulder": {"x": "left_shoulder_X", "y": "left_shoulder_Y", "z": "left_shoulder_Z"},
    "right_shoulder": {"x": "right_shoulder_X", "y": "right_shoulder_Y", "z": "right_shoulder_Z"},
    "left_elbow": {"x": "left_elbow_X", "y": "left_elbow_Y", "z": "left_elbow_Z"},
    "right_elbow": {"x": "right_elbow_X", "y": "right_elbow_Y", "z": "right_elbow_Z"},
    "left_wrist": {"x": "left_wrist_X", "y": "left_wrist_Y", "z": "left_wrist_Z"},
    "right_wrist": {"x": "right_wrist_X", "y": "right_wrist_Y", "z": "right_wrist_Z"},
    "left_hip": {"x": "left_hip_X", "y": "left_hip_Y", "z": "left_hip_Z"},
    "right_hip": {"x": "right_hip_X", "y": "right_hip_Y", "z": "right_hip_Z"},
    "left_knee": {"x": "left_knee_X", "y": "left_knee_Y", "z": "left_knee_Z"},
    "right_knee": {"x": "right_knee_X", "y": "right_knee_Y", "z": "right_knee_Z"},
    "left_ankle": {"x": "left_ankle_X", "y": "left_ankle_Y", "z": "left_ankle_Z"},
    "right_ankle": {"x": "right_ankle_X", "y": "right_ankle_Y", "z": "right_ankle_Z"},
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

# 左半身(3D)モデル用
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

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠ 読み込みエラー: {e}")
        continue

    # ROMBERGの3Dデータでは右半身が不安定なため、常に左半身モデルを使用
    req_joints = [
        "left_ear_X", "left_ear_Y", "left_ear_Z",
        "left_shoulder_X", "left_shoulder_Y", "left_shoulder_Z",
        "left_elbow_X", "left_elbow_Y", "left_elbow_Z",
        "left_wrist_X", "left_wrist_Y", "left_wrist_Z",
        "left_hip_X", "left_hip_Y", "left_hip_Z",
        "left_knee_X", "left_knee_Y", "left_knee_Z",
        "left_ankle_X", "left_ankle_Y", "left_ankle_Z",
    ]

    seg_pairs = SEGMENT_PAIRS_LEFT
    seg_others = SEGMENT_OTHERS_LEFT
    mass_ratio = BODY_SEGMENTS_MASS_RATIO_LEFT
    model_name = "左半身モデル/3D_ROMBERG"


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
    df["CoG_Z"] = 0.0
    total_mass = sum(mass_ratio.values())

    # 1. 四肢
    for segment, (prox_name, dist_name) in seg_pairs.items():
        if JOINTS[prox_name]["x"] not in df.columns or JOINTS[dist_name]["x"] not in df.columns:
            continue
        
        ratio_key = segment.replace("left_", "").replace("right_", "")
        ratio = COM_PROXIMAL_RATIOS.get(ratio_key, 0.5)

        prox_x = df[JOINTS[prox_name]["x"]]
        prox_y = df[JOINTS[prox_name]["y"]]
        prox_z = df[JOINTS[prox_name]["z"]]

        dist_x = df[JOINTS[dist_name]["x"]]
        dist_y = df[JOINTS[dist_name]["y"]]
        dist_z = df[JOINTS[dist_name]["z"]]

        cx = prox_x + (dist_x - prox_x) * ratio
        cy = prox_y + (dist_y - prox_y) * ratio
        cz = prox_z + (dist_z - prox_z) * ratio

        m = mass_ratio[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m
        df["CoG_Z"] += cz * m

    # 2. その他
    for segment, landmarks in seg_others.items():
        valid = [name for name in landmarks if JOINTS[name]["x"] in df.columns]
        if not valid:
            continue

        cx = sum(df[JOINTS[n]["x"]] for n in valid) / len(valid)
        cy = sum(df[JOINTS[n]["y"]] for n in valid) / len(valid)
        cz = sum(df[JOINTS[n]["z"]] for n in valid) / len(valid)

        m = mass_ratio[segment]
        df["CoG_X"] += cx * m
        df["CoG_Y"] += cy * m
        df["CoG_Z"] += cz * m

    # 総質量で割る（全身比率の合計が1.0にならないケースや半身モデルの場合への対応）
    df["CoG_X"] /= total_mass
    df["CoG_Y"] /= total_mass
    df["CoG_Z"] /= total_mass

    # --- スケール変換（m → mm） ---
    df["CoG_X_mm"] = df["CoG_X"] * 1000
    df["CoG_Y_mm"] = df["CoG_Y"] * 1000
    df["CoG_Z_mm"] = df["CoG_Z"] * 1000

    # --- 出力CSV ---
    out_csv = os.path.join(out_subdir, f"{base_name}_CoG.csv")
    df[["TIME", "CoG_X_mm", "CoG_Y_mm", "CoG_Z_mm"]].to_csv(out_csv, index=False)
    print(f"  ✅ 出力: {out_csv}")

    # --- プロット ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
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

    axes[2].plot(t, df["CoG_Z_mm"], color="b", label="Z軸 [mm]")
    axes[2].set_ylabel("Z [mm]")
    axes[2].set_xlabel("時間 [ms]")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    
    out_plot = os.path.join(out_subdir, f"{base_name}_CoG.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_plot)
    plt.close(fig)
    print(f"  📈 グラフ保存: {out_plot}")

print("\n=== 全ファイルの処理が完了しました ===")