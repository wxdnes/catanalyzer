"""
prepare_dataset.py
==================
โหลดวิดีโอแมวจาก YouTube URL และตัดเป็นคลิปสั้นๆ สำหรับ train model

ขั้นตอน:
1. ใส่ YouTube URL ที่ต้องการใน VIDEO_URLS ด้านล่าง
2. รัน script — จะโหลดและตัดคลิปให้อัตโนมัติ
3. จัดเข้าโฟลเดอร์ dataset/playful, lazy, shy

ติดตั้งก่อนใช้:
    pip install yt-dlp opencv-python
"""

import os
import subprocess
import cv2

# ============================================================
# ตั้งค่า: แก้ตรงนี้ได้เลย
# ============================================================

CLIP_DURATION = 10      # ความยาวแต่ละคลิป (วินาที)
CLIPS_PER_VIDEO = 5     # ตัดกี่คลิปต่อวิดีโอ
SKIP_START = 10         # ข้ามกี่วินาทีแรก (มักเป็น intro)
TARGET_CLIPS = 20       # เป้าหมายกี่คลิปต่อ class

# ใส่ YouTube URL ที่เลือกเองตรงนี้
# หาวิดีโอที่แน่ใจว่าเป็นแมวซน/ขี้เกียจ/ขี้อายจริงๆ
VIDEO_URLS = {
    "playful": [
        "https://youtube.com/shorts/aQrPm60cpB0?si=zabyGanDMHIE9OsD",
        "https://youtube.com/shorts/0fYmbaj7m0E?si=BBb9ShOLYTZo0JE9",
        "https://www.youtube.com/watch?v=0SD2CGp8ekQ",
        "https://www.youtube.com/watch?v=9FjGP4t2zKY"
    ],
    "lazy": [
        "https://youtube.com/shorts/1hG_PFkDCaA?si=mGD5Zy-oYd_pJ9Ai",
        "https://youtube.com/shorts/9HEZ2Mtc_iQ?si=LR8nMr24ENZrSQjs"
    ],
    "shy": [
        "https://www.youtube.com/shorts/8c0ih0AAxg8",
    ],
}

# ============================================================
# ส่วนที่ 1: โหลดวิดีโอจาก YouTube URL
# ============================================================

def download_videos(class_name: str, urls: list, output_dir: str):
    """โหลดวิดีโอจาก YouTube URL ที่กำหนด"""
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    if not urls:
        print(f"\n⚠️  class '{class_name}' ยังไม่มี URL — ข้ามไป")
        return []

    print(f"\n📥 กำลังโหลดวิดีโอ class '{class_name}' ({len(urls)} URLs)...")

    for url in urls:
        cmd = [
            "yt-dlp",
            url,
            "-o", os.path.join(raw_dir, "%(id)s.%(ext)s"),
            "--format", "mp4/bestvideo[ext=mp4]",
            "--max-filesize", "200M",
            "--match-filter", "duration < 600",  # ไม่เกิน 10 นาที
            "--no-playlist",
        ]
        print(f"  ⬇️  {url}")
        try:
            subprocess.run(cmd, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  timeout — ข้ามไป")
        except FileNotFoundError:
            print("  ❌ ไม่พบ yt-dlp กรุณารัน: pip install yt-dlp")
            return []

    videos = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.endswith((".mp4", ".mkv", ".webm"))
    ]
    print(f"  โหลดได้ {len(videos)} วิดีโอ")
    return videos


# ============================================================
# ส่วนที่ 2: ตัดวิดีโอเป็นคลิปสั้น
# ============================================================

def split_video(video_path: str, output_dir: str, class_name: str,
                clip_duration: int = 10, clips_per_video: int = 5,
                skip_start: int = 10) -> int:
    """
    ตัดวิดีโอยาวๆ เป็นคลิปสั้น
    return: จำนวนคลิปที่ตัดได้
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠️  เปิดไม่ได้: {os.path.basename(video_path)}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    cap.release()

    if total_duration < clip_duration + skip_start:
        print(f"  ⚠️  วิดีโอสั้นเกิน ({total_duration:.0f}s) — ข้าม")
        return 0

    # คำนวณจุดเริ่มต้นของแต่ละคลิป
    usable_duration = total_duration - skip_start - clip_duration
    if usable_duration <= 0:
        return 0

    interval = usable_duration / clips_per_video
    count = 0

    # นับคลิปที่มีอยู่แล้ว
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".mp4")])

    for i in range(clips_per_video):
        start_sec = skip_start + i * interval
        clip_name = f"{class_name}_{existing + count:03d}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        cmd = [
            "ffmpeg",
            "-ss", str(start_sec),
            "-i", video_path,
            "-t", str(clip_duration),
            "-c:v", "libx264",
            "-an",                  # ไม่เอา audio (ไม่จำเป็น)
            "-vf", "scale=320:240", # resize ให้พอดี
            "-y",                   # overwrite ถ้ามีอยู่แล้ว
            "-loglevel", "quiet",
            clip_path,
        ]

        try:
            result = subprocess.run(cmd, timeout=60)
            if result.returncode == 0 and os.path.exists(clip_path):
                count += 1
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return count


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 50)
    print("  Cat Dataset Preparer")
    print("=" * 50)

    # เช็ค ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    except FileNotFoundError:
        print("\n❌ ไม่พบ ffmpeg!")
        print("   Windows: https://ffmpeg.org/download.html")
        print("   แล้วเพิ่ม ffmpeg ใน PATH")
        return

    dataset_dir = "dataset"
    total_clips = {}

    for class_name, urls in VIDEO_URLS.items():
        class_dir = os.path.join(dataset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # โหลดวิดีโอ
        videos = download_videos(class_name, urls, class_dir)

        # ตัดเป็นคลิป
        print(f"\n✂️  กำลังตัดคลิป class '{class_name}'...")
        clip_count = 0
        for video in videos:
            n = split_video(
                video, class_dir, class_name,
                clip_duration=CLIP_DURATION,
                clips_per_video=CLIPS_PER_VIDEO,
                skip_start=SKIP_START,
            )
            clip_count += n
            if clip_count >= TARGET_CLIPS:
                break

        total_clips[class_name] = clip_count
        print(f"  ✅ ได้ {clip_count} คลิป")

    # สรุป
    print("\n" + "=" * 50)
    print("สรุป dataset:")
    for cls, count in total_clips.items():
        status = "✅" if count >= TARGET_CLIPS else "⚠️ "
        print(f"  {status} {cls}: {count}/{TARGET_CLIPS} คลิป")

    all_ready = all(c >= TARGET_CLIPS for c in total_clips.values())
    if all_ready:
        print("\n🎉 Dataset พร้อมแล้ว! รัน: py trainer.py")
    else:
        print("\n⚠️  บางคลิปยังไม่ครบ ลองรันซ้ำหรือเพิ่ม search terms")


if __name__ == "__main__":
    main()