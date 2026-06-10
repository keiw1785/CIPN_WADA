import os, re, uuid, tempfile, subprocess, sys
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from natsort import natsorted

# ==========================================================
# === 設定エリア ===========================================
# ==========================================================
# 実際に使用しているパスに合わせて書き換えてください
TARGET_ROOT_DIR = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\0_raw\calib\CIPN\P005"

# FFmpegのパス 
FFMPEG_PATH = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\etc\ffmpeg\bin\ffmpeg.exe"
OUTPUT_ROOT = r"C:\Users\kei15\CIPN\CIPN_SUGAWARA\data\1_processed\calib_trimed"

SR, FPS, HEAD_SEC = 44100, 60, 10
tempfile.tempdir = r"C:\Temp_FFmpeg" # 必要に応じて変更
os.makedirs(tempfile.gettempdir(), exist_ok=True)

print("=== 📐 キャリブレーション動画 自動同期 (CPU処理版) ===")

# ==========================================================
# === 関数群 ===============================================
# ==========================================================
def detect_clap_time(video_path, head_sec=HEAD_SEC):
    """音声解析：拍手タイミングの検出"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {video_path}")
        
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
    
    duration = get_duration(video_path)
    scan_sec = min(head_sec, duration)

    subprocess.run([
        FFMPEG_PATH, "-y", "-loglevel", "error",
        "-t", str(scan_sec), "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1", tmp
    ], check=True)

    y, sr = librosa.load(tmp, sr=None)
    os.remove(tmp)

    frame, hop = int(0.02 * sr), int(0.01 * sr)
    energy = np.array([np.sum(y[i:i+frame]**2) for i in range(0, len(y)-frame, hop)])
    clap_time = np.argmax(energy) * hop / sr
    return clap_time, y, sr

def get_duration(video_path):
    cmd = [FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe"), "-v", "error",
           "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except:
        return 0.0

def trim_and_convert(video_path, out_path, start, end):
    print(f"✂️  保存: {out_path.name}")
    # 修正箇所: h264_nvenc (GPU) -> libx264 (CPU) に変更
    # -crf 18 は画質設定（数値が低いほど高画質、18-23が標準）
    subprocess.run([
        FFMPEG_PATH, "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", str(video_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", str(out_path)
    ], check=True)

def plot_wave_check(y1, y2, sr, offset, title, png_path):
    t = np.arange(min(len(y1), len(y2))) / sr
    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    
    # 補正前
    ax[0].plot(t, y1, label="C1")
    ax[0].plot(t, y2, label="C2", alpha=0.6)
    ax[0].set_title(f"[{title}] Before Sync")
    
    # 補正後
    shift_samples = int(-offset * sr)
    y2_shifted = np.roll(y2, shift_samples)
    ax[1].plot(t, y1, label="C1")
    ax[1].plot(t, y2_shifted, label="C2 shifted", alpha=0.6)
    ax[1].set_title(f"After Sync (offset={offset:+.3f}s)")
    
    for a in ax:
        a.grid(True); a.legend()
    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

# ==========================================================
# === メイン処理 ===========================================
# ==========================================================
def process_calibration_videos():
    root = Path(TARGET_ROOT_DIR)
    dir_c1 = root / "C1"
    dir_c2 = root / "C2"

    # --- フォルダ名から命名情報を取得 ---
    pid = root.name          # P002
    group = root.parent.name # CIPN
    
    # ソートして取得
    files_c1 = natsorted([f for f in dir_c1.glob("*.MOV")])
    files_c2 = natsorted([f for f in dir_c2.glob("*.MOV")])

    print(f"📂 対象パス: {root}")
    print(f"📛 識別情報: Group={group}, ID={pid}")
    print(f"🎬 ファイル数: C1={len(files_c1)}, C2={len(files_c2)}")

    settings = [ "Setting1", "Setting2"]
    
    for i, setting_name in enumerate(settings):
        if i >= len(files_c1) or i >= len(files_c2):
            print(f"⚠️ {setting_name} 用の動画ペアが不足しています。スキップします。")
            continue

        p1 = files_c1[i]
        p2 = files_c2[i]
        
        print(f"\n──────────────────────────────")
        print(f"▶ {group} {pid} {setting_name} 処理開始")

        try:
            # 1. 拍手検出
            clap1, y1, sr1 = detect_clap_time(p1)
            clap2, y2, sr2 = detect_clap_time(p2)
            
            # 2. トリミング範囲計算
            offset = clap2 - clap1
            dur1 = get_duration(p1)
            dur2 = get_duration(p2)
            
            remain1 = dur1 - clap1
            remain2 = dur2 - clap2
            valid_duration = min(remain1, remain2)

            s1, e1 = clap1, clap1 + valid_duration
            s2, e2 = clap2, clap2 + valid_duration

            # 3. 出力設定
            out_dir = Path(OUTPUT_ROOT) / group / pid / setting_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path1 = out_dir / f"{group}_{pid}_C1_{setting_name}.mp4"
            out_path2 = out_dir / f"{group}_{pid}_C2_{setting_name}.mp4"
            wave_path = out_dir / f"{group}_{pid}_{setting_name}_wave.png"

            # 実行
            trim_and_convert(p1, out_path1, s1, e1)
            trim_and_convert(p2, out_path2, s2, e2)
            plot_wave_check(y1, y2, sr1, offset, setting_name, wave_path)
            
            print(f"✅ 完了")

        except Exception as e:
            print(f"❌ 失敗: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 全処理終了")

if __name__ == "__main__":
    process_calibration_videos()