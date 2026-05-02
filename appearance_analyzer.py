"""
appearance_analyzer.py
=======================
วิเคราะห์ลักษณะภายนอกของแมว: สี ขน รูปร่าง
ใช้ K-Means clustering และ texture analysis
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")


def load_frames(video_path: str, fps: float = 2.0) -> list:
    """โหลด frames จากวิดีโอ"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps / fps))

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(cv2.resize(frame, (320, 240)))
        count += 1

    cap.release()
    return frames


def get_dominant_colors(frame: np.ndarray, n_colors: int = 2) -> list:
    """
    หาสีหลักของแมวใน frame ด้วย K-Means clustering
    return: list ของ hex color strings เรียงตาม dominant มากสุด
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ตัดขอบออก เน้นกลางภาพ
    h, w = rgb.shape[:2]
    mh, mw = h // 6, w // 6
    cropped = rgb[mh:h - mh, mw:w - mw]

    pixels = cropped.reshape(-1, 3).astype(np.float32)

    # กรอง pixel ที่ใกล้ขาวหรือดำ (น่าจะเป็นพื้นหลัง)
    brightness = pixels.mean(axis=1)
    mask = (brightness > 20) & (brightness < 235)
    filtered = pixels[mask]

    if len(filtered) < n_colors * 10:
        filtered = pixels

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=5)
    kmeans.fit(filtered)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)

    colors = []
    for idx in sorted_idx:
        r, g, b = kmeans.cluster_centers_[idx].astype(int)
        colors.append(f"#{r:02X}{g:02X}{b:02X}")

    return colors


def estimate_fur_length(frame: np.ndarray) -> str:
    """
    ประเมินความยาวขนจาก texture variance
    ขนยาว = texture variance สูง (ขนฟูฟ่อง)
    ขนสั้น = texture เรียบกว่า
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return "long" if variance > 300 else "short"


def estimate_fur_pattern(frame: np.ndarray) -> str:
    """
    ประเมินลายขน: solid (ทึบ) หรือ patterned (มีลาย)
    ดูจากการกระจายของ hue ใน frame
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    h, w = hue.shape
    m = max(h, w) // 8
    hue_crop = hue[m:h - m, m:w - m]

    std = hue_crop.std()
    return "patterned" if std > 25 else "solid"


def analyze_appearance(frames: list) -> dict:
    """
    วิเคราะห์ลักษณะภายนอกของแมวจากหลาย frames
    return: dict ของ appearance features พร้อมส่งให้ Unity
    """
    if not frames:
        return {
            "fur_color_primary": "#808080",
            "fur_color_secondary": "#A0A0A0",
            "fur_length": "short",
            "fur_pattern": "solid",
            "body_size": "medium",
            "tail_length": "medium",
        }

    all_colors_1, all_colors_2 = [], []
    fur_lengths, fur_patterns = [], []

    for frame in frames:
        colors = get_dominant_colors(frame, n_colors=2)
        if len(colors) >= 1:
            all_colors_1.append(colors[0])
        if len(colors) >= 2:
            all_colors_2.append(colors[1])
        fur_lengths.append(estimate_fur_length(frame))
        fur_patterns.append(estimate_fur_pattern(frame))

    def most_common(lst):
        return max(set(lst), key=lst.count) if lst else "unknown"

    def avg_hex_color(hex_list):
        if not hex_list:
            return "#808080"
        rgbs = [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]
                for h in hex_list]
        avg = np.mean(rgbs, axis=0).astype(int)
        return f"#{avg[0]:02X}{avg[1]:02X}{avg[2]:02X}"

    # ประเมิน body_size จาก contour area
    size_ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        ratio = np.count_nonzero(thresh) / (frame.shape[0] * frame.shape[1])
        size_ratios.append(ratio)

    avg_size = np.mean(size_ratios)
    if avg_size > 0.6:
        body_size = "large"
    elif avg_size > 0.35:
        body_size = "medium"
    else:
        body_size = "small"

    # ประเมิน tail_length จาก aspect ratio
    ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            ratios.append(bw / bh if bh > 0 else 1.0)

    if ratios:
        avg_ratio = np.mean(ratios)
        tail_length = "long" if avg_ratio > 1.4 else ("medium" if avg_ratio > 0.9 else "short")
    else:
        tail_length = "medium"

    return {
        "fur_color_primary": avg_hex_color(all_colors_1),
        "fur_color_secondary": avg_hex_color(all_colors_2),
        "fur_length": most_common(fur_lengths),
        "fur_pattern": most_common(fur_patterns),
        "body_size": body_size,
        "tail_length": tail_length,
    }


if __name__ == "__main__":
    print("=== ทดสอบ appearance_analyzer ===\n")

    test_cases = [
        ("แมวสีส้ม", (30, 120, 200)),
        ("แมวสีขาว", (230, 230, 230)),
        ("แมวสีเทา", (120, 120, 120)),
    ]

    for name, color in test_cases:
        frames = []
        for _ in range(5):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :] = color
            noise = np.random.randint(-15, 15, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)

        result = analyze_appearance(frames)
        print(f"{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print()

    print("สำเร็จ!")
