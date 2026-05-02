"""
api.py
======
Flask API server สำหรับรับวิดีโอแมวและส่ง JSON กลับให้ Unity
"""

import os
import tempfile
from flask import Flask, request, jsonify
from trainer import predict_personality, load_model
from appearance_analyzer import analyze_appearance, load_frames

app = Flask(__name__)
MODEL_DIR = "models"
_model_cache = None


def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model(MODEL_DIR)
    return _model_cache


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Cat Analyzer API พร้อมใช้งาน"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์วิดีโอ กรุณาแนบไฟล์ด้วย key 'video'"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "ชื่อไฟล์ว่าง"}), 400

    suffix = os.path.splitext(video_file.filename)[1].lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        return jsonify({"error": f"ไม่รองรับไฟล์ประเภท {suffix}"}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        # วิเคราะห์ personality ด้วย ML model
        personality = predict_personality(tmp_path, MODEL_DIR)

        # วิเคราะห์ appearance ด้วย load_frames จาก appearance_analyzer
        frames = load_frames(tmp_path, fps=2.0)
        appearance = analyze_appearance(frames) if frames else {}

        return jsonify({
            "personality": personality,
            "appearance": appearance,
            "frames_analyzed": len(frames),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("Cat Analyzer API Server")
    print("=" * 40)

    if not os.path.exists(os.path.join(MODEL_DIR, "cat_model.pkl")):
        print("ไม่พบ model! กรุณา train ก่อนโดยรัน: python trainer.py")
    else:
        _, _, info = get_model()
        print(f"โหลด model สำเร็จ — Classes: {info['classes']}")
        print("เริ่ม server ที่ http://localhost:5000")

    app.run(host="0.0.0.0", port=5000, debug=False)
