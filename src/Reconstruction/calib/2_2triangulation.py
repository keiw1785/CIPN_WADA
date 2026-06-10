import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from tqdm import tqdm


# ==========================================================
# === 設定エリア ===========================================
# ==========================================================
# 1. データが入っている親フォルダ
DATA_ROOT_DIR = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\main_research\CIPN\P005"

# 2. ステレオパラメータファイルのパス
CALIB_PARAM_ROOT = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\calib_trimed\CIPN\P005" ##これにSettingとcamera_params_stereo.npzをつければ良い
##タスクは4MWALK,ROMBERG,TUGの3つで、Setting1が4MWALK,TUGでSetting2がROMBERG

# 3. 結果を出力するフォルダ
OUTPUT_ROOT_DIR = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\3D_Result\CIPN\P005"

TASK_SETTING_MAP = {
    "4MWALK": "Setting1",
    "TUG":"Setting1",
    "ROMBERG":"Setting2",
}

CALIB_FILENAME = "camera_params_stereo.npz"


# ==========================================================
# === クラス定義: 3D計算機 =================================
# ==========================================================
class StereoTriangulator:
    def __init__(self, calib_file):
        try:
            data = np.load(calib_file)
            self.mtx1 = data['mtx1']
            self.dist1 = data['dist1']
            self.mtx2 = data['mtx2']
            self.dist2 = data['dist2']
            self.R = data['R']
            self.T = data['T']
            
            # 投影行列の作成
            self.P1 = np.dot(self.mtx1, np.hstack((np.eye(3), np.zeros((3, 1)))))
            self.P2 = np.dot(self.mtx2, np.hstack((self.R, self.T)))
            print(f"✅ パラメータロード完了: {os.path.basename(calib_file)}")
        except Exception as e:
            raise Exception(f"❌ パラメータ読み込みエラー: {e}")

    def triangulate_batch(self, points_l, points_r):
        """
        複数の点をまとめて3D変換する
        points_l: shape (N, 2) array
        points_r: shape (N, 2) array
        """
        if len(points_l) == 0: return np.array([])

        # OpenCVの形式に合わせる (N, 1, 2)
        pts_l = np.array(points_l, dtype=np.float32).reshape(-1, 1, 2)
        pts_r = np.array(points_r, dtype=np.float32).reshape(-1, 1, 2)

        # 1. 歪み補正
        pts_l_undist = cv2.undistortPoints(pts_l, self.mtx1, self.dist1, P=self.mtx1)
        pts_r_undist = cv2.undistortPoints(pts_r, self.mtx2, self.dist2, P=self.mtx2)

        # 2. 三角測量
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pts_l_undist, pts_r_undist)

        # 3. 3D座標変換 (X/w, Y/w, Z/w)
        # wが0に近い場合は無限遠点として処理(念のため)
        w = points_4d[3]
        w[w == 0] = 1e-10
        points_3d = points_4d[:3] / w
        
        return points_3d.T # Shape (N, 3)
    
def remove_jumps_and_interpolate_3d(df_3d, threshold=0.2, limit=5):
    """
    3D化後の座標列について、縦方向に見て飛び値をNaNにし、
    線形補間する。

    threshold:
        前後平均との差がこの値以上なら飛び値とみなす
    limit:
        連続して補間する最大フレーム数
    """
    df_fixed = df_3d.copy()

    coord_cols = [
        c for c in df_fixed.columns
        if c.endswith("_X") or c.endswith("_Y") or c.endswith("_Z")
    ]

    for col in coord_cols:
        s = pd.to_numeric(df_fixed[col], errors="coerce")

        # 前との差
        diff_prev = (s - s.shift(1)).abs()

        # 1点だけ尖っている値を飛び値とする
        jump_mask = (
            (diff_prev >= threshold)
        )
        #print(col, "補間対象数:", jump_mask.sum())

        df_fixed.loc[jump_mask, col] = np.nan

    # 線形補間
    df_fixed[coord_cols] = df_fixed[coord_cols].interpolate(
        method="linear",
        limit=limit,
        limit_direction="both"
    )

    return df_fixed

# ==========================================================
# === メイン処理ロジック ===================================
# ==========================================================
def process_csv_pair(csv_c1, csv_c2, triangulator, output_path):
    # CSV読み込み
    df1 = pd.read_csv(csv_c1)
    df2 = pd.read_csv(csv_c2)

    # 行数が違う場合、短い方に合わせる
    min_len = min(len(df1), len(df2))
    df1 = df1.iloc[:min_len]
    df2 = df2.iloc[:min_len]

    # 結果を格納するDataFrame (TIME列があれば保持)
    df_3d = pd.DataFrame()
    if 'TIME' in df1.columns:
        df_3d['TIME'] = df1['TIME']
    else:
        df_3d['FRAME'] = range(min_len)

    # カラム名から関節部位を特定 ("_X" で終わる列を探す)
    # 例: "nose_X" -> body_part = "nose"
    cols = [c for c in df1.columns if c.endswith('_X')]
    body_parts = [c[:-2] for c in cols] # "_X" を除去

    print(f"   関節数: {len(body_parts)}箇所, フレーム数: {min_len}")

    for part in body_parts:
        col_x = f"{part}_X"
        col_y = f"{part}_Y"

        # 列が存在するか確認
        if col_x not in df1.columns or col_y not in df1.columns:
            continue

        # 座標データの抽出
        pts_l = df1[[col_x, col_y]].values
        pts_r = df2[[col_x, col_y]].values
        
        # 3D計算実行
        pts_3d = triangulator.triangulate_batch(pts_l, pts_r)

        # 異常値除去: 座標が(0,0)の点は計算不能としてNaNにする
        mask = (pts_l[:,0] == 0) | (pts_r[:,0] == 0)
        pts_3d[mask] = np.nan

        # 結果をDataFrameに追加 (X, Y, Z)
        df_3d[f"{part}_X"] = pts_3d[:, 0]
        df_3d[f"{part}_Y"] = pts_3d[:, 1]
        df_3d[f"{part}_Z"] = pts_3d[:, 2]

    #飛び値除去＋線形補間
    df_3d = remove_jumps_and_interpolate_3d(
        df_3d,
        threshold=0.2,
        limit=5
    )

    # 保存
    df_3d.to_csv(output_path, index=False)
    print(f"   💾 保存完了: {os.path.basename(output_path)}")


def main():
    root_path = Path(DATA_ROOT_DIR)
    output_root_path = Path(OUTPUT_ROOT_DIR)
    triangulators = {}
    
    # C1フォルダ内のCSVを探す
    c1_files = list(root_path.rglob("C1/*.csv"))

    print(f"📂 処理対象ファイル数: {len(c1_files)}")

    for file_c1 in tqdm(c1_files):
        #rootからの相対パス
        relative_path = file_c1.parent.relative_to(root_path) # 例: 4MWALK\C1
        task_name = relative_path.parts[0] # 例: 4MWALK

        if task_name not in TASK_SETTING_MAP:
            print(f"Setting未定義のタスクです。スキップ：{task_name}")
            continue

        setting = TASK_SETTING_MAP[task_name]

        calib_param_path = Path(CALIB_PARAM_ROOT)/ setting / CALIB_FILENAME

        if not calib_param_path.exists():
            print(f"❌ パラメータファイルがありません: {calib_param_path}")
            continue

        if setting not in triangulators:
            triangulators[setting] = StereoTriangulator(calib_param_path)
        
        triangulator = triangulators[setting]


        # C2ファイルのパスを推測
        # 親フォルダ名置換: .../C1/... -> .../C2/...
        dir_c2 = str(file_c1.parent).replace("C1", "C2")
        
        # ファイル名置換: ..._C1_... -> ..._C2_...
        # ハイフン区切りとアンダースコア区切りの両方に対応
        name_c2 = file_c1.name.replace("_C1_", "_C2_").replace("-C1_", "-C2_").replace("_C1", "_C2")
        
        file_c2 = Path(dir_c2) / name_c2

        if not file_c2.exists():
            print(f"⚠️ ペアが見つかりません (スキップ): {file_c1.name}")
            continue

        # 出力先のパス作成
        # 元のタスクフォルダ名 (4MWALK, ONELEG等) を維持
        save_dir = output_root_path / task_name
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            
        # 出力ファイル名 (..._3D_.csv)
        save_filename = file_c1.name.replace("_C1_", "_3D_").replace("-C1_", "-3D_")
        save_path = save_dir / save_filename

        try:
            process_csv_pair(file_c1, file_c2, triangulator, save_path)
        except Exception as e:
            print(f"❌ エラー発生 ({file_c1.name}): {e}")

if __name__ == "__main__":
    main()