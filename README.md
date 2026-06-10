# CIPN_SUGAWARA (姿勢推定解析パイプライン)

CIPN患者の身体機能評価タスクを評価するためのコードです．
研究室のパソコン上でViTPoseを使って推定したキーポイントの座標について解析を行います．

---

## 🛠 環境構築とデータ取得

<details>
<summary>コードのクローンと環境構築</summary>

本プロジェクトを実行するためのローカル環境構築手順です．

### 1．リポジトリのクローン
GitHubからプロジェクトをローカル環境にダウンロードします．

   git clone [https://github.com/41u1/CIPN_SUGAWARA.git](https://github.com/41u1/CIPN_SUGAWARA.git)

### 2．プロジェクトディレクトリへの移動
クローンして作成されたフォルダ内に移動します．

    cd <プロジェクト名>

### 3．仮想環境（venv）の作成
プロジェクト専用のPython仮想環境を作成します．これにより，他のプロジェクトとのパッケージの競合を防ぎます．

    python -m venv venv

※ 最後の `venv` は仮想環境のフォルダ名です．必要に応じて `.venv` などに変更してください．

### 4．仮想環境の有効化
作成した仮想環境をアクティベートします．お使いのOSに合わせて以下のコマンドを実行してください．

**Windowsの場合:**

    .\venv\Scripts\activate

※ PowerShellを使用していて「スクリプトの実行がシステムで無効になっている」というエラーが出る場合は，一度 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` を実行してから再度アクティベートをお試しください．

**macOS / Linuxの場合:**

    source venv/bin/activate

成功すると，ターミナルのプロンプトの先頭に `(venv)` と表示されます．

### 5．ライブラリのインストール
仮想環境が有効になっている状態で，`requirements.txt` に記載されている依存パッケージを一括インストールします．

    pip install -r requirements.txt

---

**参考リンク:** [venvで手軽にPythonの仮想環境を構築しよう](https://qiita.com/shun_sakamoto/items/7944d0ac4d30edf91fde)

</details>

<details>
<summary>データ取得と姿勢推定の実行手順（ワークフロー）</summary>

## 🏃 姿勢推定の実行手順（ワークフロー）

研究室のPC（Windows環境）を用いて，iPadで撮影した動画からViTPoseによる姿勢推定を行い，結果のCSVを自身のPCへ移行するまでの流れです．

### 1．動画データの保存（iPadからの転送）
iPadを直接Type-Cケーブルで研究室のPCに接続し，動画データを転送します．

1. 以下のディレクトリに移動します．
   `D:/PythonProject_Sugawara/vitpose/Data/raw/`
2. `CIPN` または `NOCIPN` のフォルダを選択し，該当する被験者番号（例: `P002`）のフォルダを開きます．
3. 撮影した動画を以下のルールで配置します．カメラの位置関係は下図を参照してください．
   * **側面カメラの動画**: `C1` フォルダに保存
   * **正面カメラの動画**: `C2` フォルダに保存

<div align="center">
<table bgcolor="white">
  <tr>
    <td align="center">
      <img width="250" alt="TUG概略図" src="https://github.com/user-attachments/assets/bb4ad52b-59d3-4085-b867-7bd424380213" />
    </td>
    <td align="center">
      <img width="250" alt="ロンベルグ概略図" src="https://github.com/user-attachments/assets/963d58b3-399f-4544-a0e2-7882905200bb" />
    </td>
  </tr>
</table>
<sup>カメラの位置関係（左：TUG，右：ロンベルグ試験）</sup>
</div>

### 2．動画データの前処理（リネーム・同期・トリミング）
音声波形を利用してカメラ間の同期とトリミングを行います．

1. 以下の前処理スクリプトを実行します．
   `D:/PythonProject_Sugawara/vitpose/src/データ前処理/片足立ちなし.py`
   ※ 実施したタスクの内容や順番を変更した場合は，コードを適宜修正してください．（現在は，撮影した時間順に TUG，4MWALK，ROMBERG とリネームする仕様になっています）
2. 実行時にプロンプトが表示されるので，手順1で動画を保存した「フォルダのパス」と「日付（例：`20260225`）」を入力します．
3. 処理が完了すると，以下の出力フォルダに結果が保存されます．
   `D:/PythonProject_Sugawara/vitpose/output/synced_trimmed_CORR/`
4. 同フォルダ内に出力される `wave_plot_summary.pdf` を開き，音声波形が正しく同期できているか（ズレがないか）を目視で確認します．

### 3．姿勢推定の実行（ViTPose）
前処理が完了した動画をViTPoseのパイプラインに読み込ませて姿勢推定を行います．

1. 前処理済み動画の入った被験者番号のフォルダ（手順2の出力先）をコピーします．
2. 以下のディレクトリ内の `CIPN` または `NOCIPN` フォルダにペーストします．
   `D:/PythonProject_Sugawara/vitpose_pipe/vitpose/Data/movie/`
3. 以下の設定ファイルを開き，ペーストしたパス名（グループ名と被験者番号）に合わせて中身を書き換えて保存します．
   * **対象ファイル**: `D:/PythonProject_Sugawara/vitpose_pipe/vitpose/src/config.py`
   * **変更箇所**: `MOVIE_DIR`，`OUTPUT_MOVIE_DIR`，`OUTPUT_RAW_DIR`
4. **コマンドプロンプト**を起動します．（※VS CodeのPowerShellではエラーが出る場合があるため，必ずコマンドプロンプトを使用してください）
5. 仮想環境 `(vitpose_env)` がアクティブになっていることを確認します．
6. 以下のコマンドでスクリプトのディレクトリに移動し，実行します．

        cd D:/PythonProject_Sugawara/vitpose_pipe/vitpose/src
        python pip_all.py

### 4．推定結果（CSV）の自身のPCへの移行

1. 姿勢推定が完了すると，以下のフォルダ内にグループ別（CIPN，NOCIPN，STUDENT，STUDENT_WALK）で結果が出力されます．
   `D:/PythonProject_Sugawara/vitpose_pipe/vitpose/output/raw/`
2. 出力されたCSVデータをUSBメモリやケーブルを利用して自身のPCにコピーし，本解析プロジェクトの以下のディレクトリに保存してください．
   `./data/1_processed/`

</details>
</details>

---

## 📁 フォルダ構成と役割

<details>
<summary>1. data/</summary>
<pre>
0_raw/                  実験で撮った元の動画データ（編集しない原本）．これは患者情報のためオフライン研究室PCで保存．
1_processed/            MediaPipeやViTPoseの出力（キーポイントCSV・動画）．姿勢推定後のCSVをダウンロードしてここに保存．
2_time_series_feature/  時系列特徴量（歩行軌跡・関節角度・重心など）の保存．
3_summary_feature/      要約特徴量（歩幅，速度などの統計量やスカラー値）の保存．
</pre>
</details>

<details>
<summary>2. src/ （ソースコードの詳細）</summary>

**【タスク別の色分け凡例】**
🟢 ロンベルグ試験　🔵 3次元化＋3次元解析　🟡 TUG試験　🔴 4m歩行

<pre>
1_preprocessing/        姿勢推定や動画前処理（MediaPipe，ViTPoseなど）．
  （※各種前処理スクリプト）

2_time_series_feature/  時系列特徴量の計算（角度・重心・歩行軌跡など）．
  ├─ 🔵 3D/
  │   ├─ 🔵 ANGLE.py
  │   └─ 🔵 GoB.py
  └─ 🟢 ROMBERG/
      ├─ 🟢 CoM_calib.py
      └─ 🟢 CoM_統合版.py

3_summary_feature/      要約特徴量の計算・抽出．
  ├─ 🔴 4MWALK/
  │   ├─ 🔴 step_detection.py
  │   └─ 🔴 箱ひげ図.py
  ├─ 🟢 ROMBERG/
  │   ├─ 🟢 narrow_window_variance.py
  │   └─ 🟢 ROMBERG_ratio_filtered.py
  ├─ 🟡 TUG/
  │   └─ 🟡 指標算出.py
  └─ その他/

4_analysis/             PCA・クラスタリング・グラフ描画など分析処理．
  ├─ 🔴 4MWALK/
  │   
  ├─ 🟢 ROMBERG/
  │   ├─ 🟢 クロフォードテスト_軌跡.py
  │   ├─ 🟢 クロフォードテスト_分散.py
  │   ├─ 🟢 重症度plot.py
  │   └─ 🟢 重心軌跡比較.py
  └─ 🟡 TUG/
      ├─ 🟡 kiseki.py
      ├─ 🟡 レーダー＋凡例.py
      └─ 🟡 多角的分析.py

etc/                    その他のスクリプト（3Dポーズ再構築やユーティリティなど）．
  └─ 🔵 Reconstruction/
      └─ 🔵 calib/
          ├─ 🔵 0_trim_calib.py
          ├─ 🔵 1_calib.py
          ├─ 🔵 2_triangulation.py
          ├─ 🔵 3_3d_visual.py
          └─ 🔵 calib_new.py
</pre>
</details>

<details>
<summary>3. configs/</summary>
<pre>
paths.yaml              データ・結果フォルダのパス設定（例：./daily_results/results_{date}）．
model_params.yaml       モデルの設定値（入力サイズ・バッチサイズなど）．
preprocessing.yaml      前処理パラメータ（フィルタ設定など）．
</pre>
</details>

<details>
<summary>4. その他</summary>
<pre>
requirements.txt        使用ライブラリ一覧（再現性確保）．
README.md               プロジェクト概要と手順．
.vscode/settings.json   VS Codeの設定（整形・仮想環境指定など）．
.vscode/launch.json     デバッグ実行設定．
</pre>
</details>

---

## 🔬 各試験の解析スクリプト詳細


<details open>
<summary>🟢 ロンベルグ試験</summary>

### 重心座標の算出
- `src\2_time_series_feature\ROMBERG\GoB_比率考慮版.py`
  - カメラC1，C2の両方について重心の座標を算出します．
  - **使い方**: 被験者番号と被験者グループを指定して実行します．
- `src\2_time_series_feature\ROMBERG\GoB_L_only.py`
  - ロンベルグ試験のC1用スクリプトです．右側が映っておらずガクガクしてしまうため，左側だけで算出して置き換えます．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/456d52bd-2d0e-471d-91d2-a87c8e7f6a9c" />
</div>

### 軌跡長・凸包面積
- `src\3_summary_feature\ROMBERG\ROMBERG_ratio_filtered.py`
  - 重心座標から2次元平面への復元を行います．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/983ca2d1-7278-4918-b0dc-bb68214dc9d6" />
</div>

- `src\4_analysis\ROMBERG\クロフォードテスト_軌跡.py`
  - CSVデータを読み込んでEO（開眼），EC（閉眼），ROMBERG Ratioの3つを箱ひげ図として出力します．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/d084036a-ee51-42b4-858e-548fe62d6ed3" />
</div>

- `src\4_analysis\ROMBERG\重心軌跡比較.py`
  - スライド用の重心軌跡の可視化図の比較用スクリプトです．表示する患者の試行を選択できます．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/3e5e5002-abf4-46b5-825c-0cb814458d35" />
</div>

### 分散
- `src\3_summary_feature\ROMBERG\narrow_window_variance.py`
  - 3つの分散（TOTAL，GLOBAL，LOCAL）を出力します．基本的にはTOTAL（一般的な分散）を用います．
- `src\4_analysis\ROMBERG\クロフォードテスト_分散.py`
  - CSVデータを読み込んでEO，EC，ROMBERG Ratioの3つを箱ひげ図として出力します．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/8dcc89ea-6a34-4574-9710-d1c9fe51812f" />
</div>

### 重症度比較
- `src\4_analysis\重症度plot.py`
  - 重症度指標とロンベルグ率との比較を行うコードです．入力は手動で行います．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/4fa8ce31-08e1-4bd8-8e67-d65bb091c25c" />
</div>

</details>

<details>
<summary>🔵 3次元化＋3次元解析 </summary>

### 3次元復元
- `src\reconstruction\calib\0_trim_calib.py`
  - 拍手音によって同期とトリミングを行います．出力は，Setting1（動的タスク）とSetting2（静的タスク）の2つです．ちゃんと同期できているかを波形で確認します．
- `src\reconstruction\calib\1_calib.py`
  - 単眼とステレオのキャリブレーションを行います．再投影誤差が十分に低いかを確認します．うまくいかない場合は `src\reconstruction\calib\calib_new.py` も試します．
- `src\reconstruction\calib\2_triangulation.py`
  - 三角測量により3次元座標を出力します．
- `src\reconstruction\calib\3_3d_visual.py`
  - 出力された座標を3次元空間にプロットします．3次元化がうまくいったかの確認とスライド用です．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/33672e38-2ed7-45ce-8b8a-644c481f91c3" />
</div>

### 3D解析（3次元化したデータを活用）
- `src\2_time_series_feature\3D\ANGLE.py`
  - 3次元の関節角度を求めます．
- `src\2_time_series_feature\3D\GoB.py`
  - 3次元の重心の座標を求めます．

</details>

<details>
<summary>🟡 TUG試験</summary>

### フェーズ分割・レーダーチャート・多角的分析
- `src\3_summary_feature\TUG\指標算出.py`
  - フェーズ分割と起立時間，旋回時間，旋回半径，着席時間，総秒数，総歩数，歩幅の比を算出します．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/79487c26-7f43-4a4d-9d4a-7a60c843d625" />
</div>

- `src\4_analysis\TUG\レーダー＋凡例.py`
  - 7つの指標のレーダーチャートを作成します（`tug_metrics_averaged.csv` を読み込みます）．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/6acb3360-fba8-4bd1-988b-6d9d8f421f56" />
</div>

- `src\4_analysis\TUG\多角的分析.py`
  - 7つの指標についてPCA分析などの多角的な分析を行います．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/71920d01-efb2-4ffb-99bd-d0b0d8135c96" />
</div>

- `src\4_analysis\TUG\kiseki.py`
  - 腰座標の水平面上の軌跡をフェーズ分割で色分けします．分割がうまくいっているかの確認とスライド用です．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/7a8ba6c5-28fa-4b27-b3f7-93ac0c49c3a8" />
</div>

</details>

<details>
<summary>🔴 4m歩行</summary>

- `src\3_summary_feature\4MWALK\step_detection.py`
  - 接地の検出の可視化用です．処理が結構重いです．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/c120d0d6-6dd1-4d57-a6b1-59b1c7df5eca" />
</div>

- `src\3_summary_feature\4MWALK\箱ひげ図.py`
  - 一般的な歩行パラメータ（歩幅，歩隔，歩行速度，ケイデンス）を箱ひげ図にまとめます．
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/3d862b8f-fae7-44ec-b113-cdd4a994bf8c" />
</div>

</details>

## 🗑 アーカイブ
<details>
<summary>ViTPoseのコード</summary>
git@gitlab.cds.tohoku.ac.jp:neurolab/nkym_subgrp/tracking_solution_cipn/vitpose.git

[ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://arxiv.org/abs/2204.12484)

venv で仮想環境を立ててから<br>

環境構築<br>
0. nvidia-smiでcudaバージョンを確認
1. Pytorchをインストール
[公式](https://pytorch.org/get-started/locally/)でコマンドを確認．<br>
例:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```
GPUの互換性とかめちゃくちゃめんどい<br>
新しめのGPUなら割とうまくいく<br>
[torchインストールの参考1](https://pytorch.org/get-started/previous-versions/)<br>
[torchインストールの参考2](https://aitoao.com/webrary/pytorch-pip/)<br>

</details>

---
<details>

<summary>Mediapipe (現在使用していない)</summary>
git@gitlab.cds.tohoku.ac.jp:neurolab/nkym_subgrp/tracking_solution_cipn/mediapipe_basic.git

### 使い方
1. requirements.txtに記述されたライブラリを作成した仮想環境にインストール

        pip install -r requirements.txt

2. dataディレクトリまたはhand_dataディレクトリにトラッキングしたい動画を配置

3. src内でプログラムを実行

   **姿勢のトラッキング**
        
        python main_tracking.py
        
   **手のトラッキング**

        python main_hand_tracking.py

4. outputディレクトリに結果の動画およびcsvが出力される

#### 参考リンク
- **body tracking**
  - [Python 用姿勢ランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python?hl=ja)
  - [姿勢ランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ja)
- **hand tracking**
  - [Python 用手のランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python?hl=ja#image)
  - [手のランドマーク検出ガイド](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index?hl=ja#models)
- **API リファレンス**
  - [Module:mp](https://ai.google.dev/edge/api/mediapipe/python/mp#classes)

</details>
