"""
feature_extractor.py (ปรับปรุงแล้ว)
=====================================
เพิ่ม temporal features เพื่อให้ Random Forest จับ pattern
การเปลี่ยนแปลงตามเวลาได้ — แก้จุดอ่อนหลักของโค้ดเดิม

การเปลี่ยนแปลงหลัก:
1. เพิ่ม temporal motion features (จับลำดับการเคลื่อนไหว)
2. เพิ่ม activity pattern (แมวขยับต้นคลิป vs ท้ายคลิป)
3. ปรับ pose features ให้ไม่พึ่ง MediaPipe (ไม่รองรับแมว)
4. ลด feature ที่ซ้ำซ้อน (motion_variance ซ้ำ motion_std)
"""

import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# ส่วนที่ 1: Motion Features (ปรับปรุง + เพิ่ม temporal)
# ============================================================

def extract_motion_features(frames: list) -> dict:
    """
    ดึง features การเคลื่อนไหว รวมถึง temporal pattern

    เพิ่มจากเดิม:
    - motion_trend      : แมวขยับขึ้นหรือลงตามเวลา (+ = ขยับมากขึ้น)
    - early_vs_late     : เปรียบ motion ครึ่งแรก vs ครึ่งหลัง
    - motion_rhythm     : วัดความสม่ำเสมอ (burst เป็นจังหวะ = เล่น)
    - rest_streak_max   : นิ่งติดกันนานสุดกี่ frame (ขี้เกียจ = นาน)
    - burst_count       : กี่ครั้งที่ขยับแรงๆ แล้วหยุด (playful = เยอะ)
    """
    if len(frames) < 2:
        return _zero_motion_features()

    motion_values = []

    for i in range(len(frames) - 1):
        prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_values.append(magnitude.mean())

    motion_arr = np.array(motion_values)
    norm = np.clip(motion_arr / 10.0, 0, 1)

    # === Temporal features (ใหม่) ===

    # แบ่งเป็นครึ่งแรกและครึ่งหลัง
    mid = len(norm) // 2
    early = norm[:mid] if mid > 0 else norm
    late = norm[mid:] if mid < len(norm) else norm

    # motion_trend: ค่าลบ = แมวเริ่มขยับแล้วหยุด (เล่นแล้วเหนื่อย)
    #               ค่าบวก = แมวเริ่มตื่นตัวขึ้น
    motion_trend = float(late.mean() - early.mean())

    # เปรียบ early vs late โดยตรง
    early_vs_late = float(early.mean() - late.mean())

    # motion_rhythm: std ของ "ช่วงที่ขยับ" แต่ละ window
    # ถ้าเล่น จะมี burst สม่ำเสมอ → rhythm ต่ำ
    # ถ้าสุ่ม จะแปรปรวนมาก → rhythm สูง
    window = max(3, len(norm) // 5)
    window_means = [norm[i:i+window].mean() for i in range(0, len(norm)-window, window)]
    motion_rhythm = float(np.std(window_means)) if len(window_means) > 1 else 0.0

    # นับ rest streak (นิ่งติดต่อกันนานสุด)
    threshold_rest = 0.05
    rest_streak_max = _max_streak(norm < threshold_rest)

    # นับ burst count (ขยับแรงแล้วกลับมานิ่ง)
    threshold_burst = 0.4
    burst_count = _count_transitions(norm, threshold_burst)

    return {
        # features เดิม (ลด variance ออก เพราะซ้ำกับ std)
        "motion_mean":      float(norm.mean()),
        "motion_max":       float(norm.max()),
        "motion_std":       float(norm.std()),
        "active_ratio":     float((norm > 0.1).mean()),
        "burst_ratio":      float((norm > 0.5).mean()),
        # features ใหม่ (temporal)
        "motion_trend":     motion_trend,
        "early_vs_late":    early_vs_late,
        "motion_rhythm":    motion_rhythm,
        "rest_streak_max":  float(rest_streak_max) / max(len(norm), 1),
        "burst_count":      float(burst_count) / max(len(norm), 1),
    }


def _max_streak(bool_arr: np.ndarray) -> int:
    """หาความยาวสูงสุดของ True ติดกัน"""
    max_s = cur = 0
    for val in bool_arr:
        cur = cur + 1 if val else 0
        max_s = max(max_s, cur)
    return max_s


def _count_transitions(norm: np.ndarray, threshold: float) -> int:
    """นับจำนวนครั้งที่ขึ้นผ่าน threshold แล้วลงกลับมา (burst pattern)"""
    above = norm > threshold
    count = 0
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:  # rising edge
            count += 1
    return count


def _zero_motion_features() -> dict:
    return {
        "motion_mean": 0.0, "motion_max": 0.0, "motion_std": 0.0,
        "active_ratio": 0.0, "burst_ratio": 0.0,
        "motion_trend": 0.0, "early_vs_late": 0.0, "motion_rhythm": 0.0,
        "rest_streak_max": 0.0, "burst_count": 0.0,
    }


# ============================================================
# ส่วนที่ 2: Pose Features (ปรับใหม่ — ไม่ใช้ MediaPipe)
# ใช้ contour analysis ที่ตรงกับแมวกว่า
# ============================================================

def extract_pose_features(frames: list) -> dict:
    """
    วิเคราะห์ท่าทางแมวจาก contour แทน MediaPipe

    เหตุผล: MediaPipe ออกแบบสำหรับคน ไม่ใช่แมว
    detection rate มักต่ำมาก ทำให้ feature ไม่มีประโยชน์

    features ใหม่:
    - body_compactness  : แมวขดตัวหรือแผ่ร่าง (ขดตัว = ขี้เกียจ/ขี้อาย)
    - aspect_ratio_mean : สัดส่วนกว้าง/สูง
    - aspect_ratio_std  : ความแปรปรวนของ pose (เปลี่ยนท่าบ่อย = ซน)
    - position_spread   : แมวอยู่กลางภาพหรือขอบ (ขอบ = ขี้อาย)
    """
    if not frames:
        return _zero_pose_features()

    compactness_list = []
    aspect_ratios = []
    center_distances = []   # ระยะจากกลางภาพ (normalize)

    h_img, w_img = frames[0].shape[:2]
    cx_img, cy_img = w_img / 2, h_img / 2

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            compactness_list.append(0.5)
            aspect_ratios.append(1.0)
            center_distances.append(0.0)
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        # compactness = 4π × area / perimeter²
        # วงกลม = 1.0, รูปยาวๆ < 1.0, แมวขดตัวจะใกล้ 1.0
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            compactness = 0.5
        compactness_list.append(min(compactness, 1.0))

        # aspect ratio
        x, y, bw, bh = cv2.boundingRect(largest)
        aspect_ratios.append(bw / bh if bh > 0 else 1.0)

        # ระยะจากกลางภาพ
        cx = x + bw / 2
        cy = y + bh / 2
        dist = np.sqrt(((cx - cx_img) / w_img) ** 2 + ((cy - cy_img) / h_img) ** 2)
        center_distances.append(dist)

    return {
        "pose_compactness_mean":   float(np.mean(compactness_list)),
        "pose_compactness_std":    float(np.std(compactness_list)),
        "pose_aspect_ratio_mean":  float(np.mean(aspect_ratios)),
        "pose_aspect_ratio_std":   float(np.std(aspect_ratios)),   # เปลี่ยนท่าบ่อย
        "pose_center_distance":    float(np.mean(center_distances)), # ขี้อาย = อยู่ขอบ
    }


def _zero_pose_features() -> dict:
    return {
        "pose_compactness_mean": 0.5, "pose_compactness_std": 0.0,
        "pose_aspect_ratio_mean": 1.0, "pose_aspect_ratio_std": 0.0,
        "pose_center_distance": 0.0,
    }


# ============================================================
# ส่วนที่ 3: Appearance Features (เหมือนเดิม)
# ============================================================

def extract_appearance_features(frames: list) -> dict:
    if not frames:
        return {f"appear_{k}": 0.0 for k in [
            "hue_mean", "hue_std", "saturation_mean",
            "brightness_mean", "texture_variance",
            "size_ratio", "color_diversity"
        ]}

    hues, saturations, brightnesses = [], [], []
    textures, size_ratios = [], []

    for frame in frames:
        h_img, w_img = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        margin = 0.15
        mh, mw = int(h_img * margin), int(w_img * margin)
        hsv_crop = hsv[mh:h_img - mh, mw:w_img - mw]

        hues.append(hsv_crop[:, :, 0].mean())
        saturations.append(hsv_crop[:, :, 1].mean() / 255.0)
        brightnesses.append(hsv_crop[:, :, 2].mean() / 255.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        textures.append(float(laplacian.var()))

        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        size_ratios.append(np.count_nonzero(thresh) / (h_img * w_img))

    color_diversity = float(np.std(hues) / 180.0)

    return {
        "appear_hue_mean":          float(np.mean(hues) / 180.0),
        "appear_hue_std":           float(np.std(hues) / 180.0),
        "appear_saturation_mean":   float(np.mean(saturations)),
        "appear_brightness_mean":   float(np.mean(brightnesses)),
        "appear_texture_variance":  float(min(np.mean(textures) / 1000.0, 1.0)),
        "appear_size_ratio":        float(np.mean(size_ratios)),
        "appear_color_diversity":   float(color_diversity),
    }


# ============================================================
# ฟังก์ชันหลัก
# ============================================================

def extract_all_features(video_path: str, sample_fps: float = 2.0):
    """
    ดึง features ทั้งหมดจากวิดีโอ
    return: (feature_vector: np.ndarray, feature_names: list)
    """
    frames = _load_frames(video_path, sample_fps)

    if not frames:
        raise ValueError(f"ไม่สามารถโหลดวิดีโอได้: {video_path}")

    motion_feats  = extract_motion_features(frames)
    pose_feats    = extract_pose_features(frames)
    appear_feats  = extract_appearance_features(frames)

    all_feats = {**motion_feats, **pose_feats, **appear_feats}

    feature_names  = sorted(all_feats.keys())
    feature_vector = np.array([all_feats[k] for k in feature_names])

    return feature_vector, feature_names


def _load_frames(video_path: str, fps: float = 2.0) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps / fps))

    frames, count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(cv2.resize(frame, (320, 240)))
        count += 1

    cap.release()
    return frames


# ============================================================
# ทดสอบ
# ============================================================

if __name__ == "__main__":
    print("=== ทดสอบ feature_extractor (ปรับปรุงแล้ว) ===\n")

    def create_test_video(path: str, motion_type: str):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
        for i in range(90):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            if motion_type == "playful":
                # ขยับเร็ว เป็น burst
                x = int(50 + (i % 12) * 16)
                cv2.rectangle(frame, (x, 60), (x + 80, 180), (30, 120, 200), -1)
            elif motion_type == "lazy":
                # นิ่งเกือบตลอด
                cv2.rectangle(frame, (110, 70), (210, 170), (200, 80, 30), -1)
            else:  # shy
                # ขยับนิดหน่อย อยู่ขอบ
                x = 10 + (i % 8) * 2
                cv2.rectangle(frame, (x, 80), (x + 60, 150), (30, 200, 120), -1)
        writer.write(frame)
        writer.release()

    import os
    videos = {
        "playful": "/tmp/test_playful.mp4",
        "lazy":    "/tmp/test_lazy.mp4",
        "shy":     "/tmp/test_shy.mp4",
    }

    for label, path in videos.items():
        create_test_video(path, label)

    print(f"{'Feature':<35} {'playful':>10} {'lazy':>10} {'shy':>10}")
    print("-" * 67)

    all_results = {}
    for label, path in videos.items():
        feats, names = extract_all_features(path)
        all_results[label] = dict(zip(names, feats))

    for name in sorted(all_results["playful"].keys()):
        row = f"{name:<35}"
        for label in ["playful", "lazy", "shy"]:
            row += f" {all_results[label][name]:>10.4f}"
        print(row)

    print(f"\nจำนวน features ทั้งหมด: {len(names)} (เดิม 16)")
    print("สำเร็จ!")