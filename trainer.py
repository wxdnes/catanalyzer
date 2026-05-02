"""
trainer.py
==========
ฝึกสอน Random Forest model จาก dataset วิดีโอแมว

Random Forest คืออะไร:
- เป็น ML algorithm ที่สร้าง "ต้นไม้ตัดสินใจ" หลายต้น (forest)
- แต่ละต้นเรียนรู้จากข้อมูลคนละชุด (random sampling)
- ตอน predict จะให้ต้นไม้ทุกต้น vote แล้วเอาเสียงข้างมาก
- เหตุที่เลือก Random Forest:
  * ทำงานดีแม้ data น้อย (60-100 ตัวอย่าง)
  * ไม่ต้องการ GPU
  * train เร็ว (ไม่กี่วินาที)
  * อธิบายได้ว่า feature ไหนสำคัญ

ขั้นตอนการทำงาน:
1. อ่านวิดีโอทุกตัวใน dataset/
2. แปลงแต่ละวิดีโอเป็น features vector
3. Train Random Forest กับ features + labels
4. ประเมินความแม่น (cross-validation)
5. บันทึก model ลงไฟล์
"""

import os
import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# import feature extractor ของเรา
from feature_extractor import extract_all_features


# ============================================================
# ส่วนที่ 1: โหลด Dataset
# ============================================================

def load_dataset(dataset_dir: str, sample_fps: float = 2.0):
    """
    อ่านวิดีโอทั้งหมดจากโฟลเดอร์ dataset และแปลงเป็น features

    โครงสร้างโฟลเดอร์ที่คาดหวัง:
    dataset/
        playful/   ← วิดีโอแมวซน
        lazy/      ← วิดีโอแมวขี้เกียจ
        shy/       ← วิดีโอแมวขี้อาย

    return:
        X: numpy array shape (n_videos, n_features) - features ทุกวิดีโอ
        y: numpy array shape (n_videos,)             - labels (playful/lazy/shy)
        feature_names: list ชื่อ features
    """
    X = []          # features ของแต่ละวิดีโอ
    y = []          # label ของแต่ละวิดีโอ
    feature_names = None
    skipped = 0

    # รองรับไฟล์วิดีโอหลายนามสกุล
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}

    # วน loop ทุก class (playful, lazy, shy)
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not classes:
        raise ValueError(f"ไม่พบโฟลเดอร์ class ใน {dataset_dir}")

    print(f"พบ {len(classes)} classes: {classes}")
    print()

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        video_files = [
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in video_extensions
        ]

        print(f"  class '{class_name}': {len(video_files)} วิดีโอ")

        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            try:
                features, names = extract_all_features(video_path, sample_fps)

                if feature_names is None:
                    feature_names = names  # บันทึก feature names จากวิดีโอแรก

                X.append(features)
                y.append(class_name)
                print(f"    ✓ {video_file}")

            except Exception as e:
                print(f"    ✗ {video_file} — ข้าม ({e})")
                skipped += 1

    print(f"\nโหลดสำเร็จ: {len(X)} วิดีโอ, ข้าม: {skipped} วิดีโอ")

    if not X:
        raise ValueError("ไม่มีวิดีโอที่โหลดได้เลย กรุณาตรวจสอบ dataset")

    return np.array(X), np.array(y), feature_names


# ============================================================
# ส่วนที่ 2: Train Model
# ============================================================

def train_model(X: np.ndarray, y: np.ndarray, feature_names: list):
    """
    Train Random Forest model

    ขั้นตอน:
    1. แปลง label string → ตัวเลข (playful=0, lazy=1, shy=2)
    2. แบ่ง data เป็น train/test (80/20)
    3. Train Random Forest
    4. ประเมินผลด้วย cross-validation
    5. แสดง feature importance

    return: (model, label_encoder, results)
    """
    print("\n=== เริ่ม Train Model ===\n")

    # แปลง label เป็นตัวเลข
    # LabelEncoder: playful→0, lazy→1, shy→2 (เรียงตาม alphabet)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {list(le.classes_)}")
    print(f"Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # แสดงการกระจายของ data
    print(f"\nการกระจาย data:")
    for cls in le.classes_:
        count = (y == cls).sum()
        print(f"  {cls}: {count} วิดีโอ")

    # แบ่ง train/test
    # test_size=0.2 หมายถึงเอา 20% เป็น test set
    # stratify=y_encoded ทำให้ทุก class มีสัดส่วนเท่ากันใน train และ test
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        print(f"\nแบ่ง data: train={len(X_train)}, test={len(X_test)}")
    else:
        # ถ้า data น้อยมาก ใช้ทั้งหมด train
        X_train, X_test = X, X
        y_train, y_test = y_encoded, y_encoded
        print(f"\ndata น้อย ({len(X)} ตัวอย่าง) ใช้ทั้งหมดเป็น train")

    # สร้างและ train Random Forest
    # n_estimators=100 = สร้างต้นไม้ 100 ต้น
    # max_depth=10 = แต่ละต้นลึกได้สูงสุด 10 ชั้น (ป้องกัน overfit)
    # min_samples_split=2 = ต้องมีอย่างน้อย 2 ตัวอย่างถึงจะแตกกิ่ง
    # random_state=42 = seed สำหรับ reproducibility
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced"  # ช่วยเมื่อ class ไม่สมดุล
    )

    print("\nกำลัง train Random Forest...")
    model.fit(X_train, y_train)
    print("Train เสร็จแล้ว!")

    # ประเมินผล
    results = _evaluate_model(model, X_train, X_test, y_train, y_test, le, X, y_encoded)

    # แสดง feature importance
    _show_feature_importance(model, feature_names)

    return model, le, results


def _evaluate_model(model, X_train, X_test, y_train, y_test, le, X_all, y_all):
    """ประเมินความแม่นของ model"""
    print("\n=== ผลการประเมิน ===\n")

    # Train accuracy
    train_acc = model.score(X_train, y_train)
    print(f"Train accuracy: {train_acc:.1%}")

    # Test accuracy
    test_acc = model.score(X_test, y_test)
    print(f"Test accuracy:  {test_acc:.1%}")

    # Cross-validation (ถ้ามี data พอ)
    # แบ่ง data เป็น 5 ส่วน train 4 ทดสอบ 1 วนไป 5 รอบ
    if len(X_all) >= 10:
        cv_scores = cross_val_score(model, X_all, y_all, cv=min(5, len(X_all) // 2))
        print(f"Cross-val accuracy: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
    else:
        cv_scores = np.array([test_acc])

    # Classification report
    y_pred = model.predict(X_test)
    print(f"\nDetailed report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }


def _show_feature_importance(model, feature_names: list):
    """แสดง feature ไหนสำคัญที่สุดสำหรับการตัดสินใจ"""
    print("\n=== Feature Importance (สำคัญมาก → น้อย) ===\n")

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for i, idx in enumerate(sorted_idx[:8]):  # แสดง 8 อันดับแรก
        bar = "█" * int(importances[idx] * 40)
        print(f"  {i+1:2d}. {feature_names[idx]:<35s} {importances[idx]:.4f} {bar}")


# ============================================================
# ส่วนที่ 3: บันทึกและโหลด Model
# ============================================================

def save_model(model, label_encoder, feature_names: list, results: dict,
               output_dir: str = "models"):
    """
    บันทึก model และข้อมูลที่เกี่ยวข้องลงไฟล์

    ไฟล์ที่บันทึก:
    - cat_model.pkl       — ตัว model จริงๆ
    - label_encoder.pkl   — ตัวแปลง label
    - model_info.json     — metadata (accuracy, feature names)
    """
    os.makedirs(output_dir, exist_ok=True)

    # บันทึก model
    model_path = os.path.join(output_dir, "cat_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # บันทึก label encoder
    le_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)

    # บันทึก metadata
    info = {
        "classes": list(label_encoder.classes_),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_estimators": model.n_estimators,
        "results": results,
    }
    info_path = os.path.join(output_dir, "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nบันทึก model สำเร็จ:")
    print(f"  {model_path}")
    print(f"  {le_path}")
    print(f"  {info_path}")


def load_model(model_dir: str = "models"):
    """โหลด model ที่ train แล้วกลับมาใช้"""
    model_path = os.path.join(model_dir, "cat_model.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    info_path = os.path.join(model_dir, "model_info.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    with open(info_path, "r") as f:
        info = json.load(f)

    return model, label_encoder, info


# ============================================================
# ส่วนที่ 4: Predict วิดีโอใหม่
# ============================================================

def predict_personality(video_path: str, model_dir: str = "models") -> dict:
    """
    รับวิดีโอแมวใหม่ → คืน personality JSON

    นี่คือฟังก์ชันที่ Flask API จะเรียกใช้
    """
    model, le, info = load_model(model_dir)

    # ดึง features จากวิดีโอ
    features, _ = extract_all_features(video_path)
    features = features.reshape(1, -1)  # reshape เป็น (1, n_features)

    # predict class และ probability
    predicted_class_idx = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # แปลง index กลับเป็น label
    predicted_label = le.inverse_transform([predicted_class_idx])[0]

    # สร้าง personality scores จาก probabilities
    # probability ของแต่ละ class = ความมั่นใจว่าเป็นนิสัยนั้น
    personality = {}
    for i, cls in enumerate(le.classes_):
        personality[cls] = round(float(probabilities[i]), 2)

    # เพิ่ม dominant personality
    personality["dominant"] = predicted_label
    personality["confidence"] = round(float(probabilities.max()), 2)

    return personality


# ============================================================
# Main: สร้าง dataset ทดสอบและ train
# ============================================================

if __name__ == "__main__":
    import cv2

    print("=== Demo: สร้าง dataset จำลองและ train model ===\n")

    # สร้างวิดีโอทดสอบ 5 ตัวต่อ class
    def make_video(path, motion_type, n_frames=90):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(path, fourcc, 30.0, (320, 240))
        for i in range(n_frames):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            if motion_type == "playful":
                x = int(50 + (i % 15) * 14)
                cv2.rectangle(frame, (x, 60), (x + 80, 180), (30, 120, 200), -1)
            elif motion_type == "lazy":
                cv2.rectangle(frame, (120, 80), (200, 160), (200, 80, 30), -1)
            else:  # shy
                x = 120 + (i % 5) * 3
                cv2.rectangle(frame, (x, 80), (x + 70, 150), (30, 200, 120), -1)
            w.write(frame)
        w.release()

    os.makedirs("dataset/playful", exist_ok=True)
    os.makedirs("dataset/lazy", exist_ok=True)
    os.makedirs("dataset/shy", exist_ok=True)

    for cls in ["playful", "lazy", "shy"]:
        for i in range(6):
            make_video(f"dataset/{cls}/cat_{i:02d}.mp4", cls)
    print("สร้าง dataset จำลองสำเร็จ (18 วิดีโอ)\n")

    # โหลด dataset
    print("กำลังโหลด dataset...")
    X, y, feature_names = load_dataset("dataset")

    # Train model
    model, le, results = train_model(X, y, feature_names)

    # บันทึก
    save_model(model, le, feature_names, results)

    print(f"\nสรุป: train accuracy = {results['train_accuracy']:.1%}")
    print("พร้อมใช้งาน!")
