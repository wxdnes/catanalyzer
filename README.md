# Cat Personality Analyzer — ML Pipeline

วิเคราะห์นิสัยแมวจากวิดีโอด้วย Machine Learning (Random Forest)

## โครงสร้างโปรเจค

```
cat_ml/
├── feature_extractor.py   ← แปลงวิดีโอ → ตัวเลข (features)
├── trainer.py             ← train Random Forest model
├── appearance_analyzer.py ← วิเคราะห์สี ขน รูปร่าง
├── api.py                 ← Flask API สำหรับ Unity
├── requirements.txt       ← library ที่ต้องติดตั้ง
├── dataset/               ← วิดีโอแมวสำหรับ train (ต้องเตรียมเอง)
│   ├── playful/           ← วิดีโอแมวซน
│   ├── lazy/              ← วิดีโอแมวขี้เกียจ
│   └── shy/               ← วิดีโอแมวขี้อาย
└── models/                ← model ที่ train แล้ว (สร้างอัตโนมัติ)
    ├── cat_model.pkl
    ├── label_encoder.pkl
    └── model_info.json
```

## ขั้นตอนการใช้งาน

### 1. ติดตั้ง library
```bash
pip install -r requirements.txt
```

### 2. เตรียม dataset
- สร้างโฟลเดอร์ dataset/playful/, dataset/lazy/, dataset/shy/
- นำวิดีโอแมว .mp4 ใส่แต่ละโฟลเดอร์ตามนิสัย
- อย่างน้อย 15-20 วิดีโอต่อ class

### 3. Train model
```bash
python trainer.py
```
ไฟล์ model จะถูกบันทึกใน models/

### 4. รัน API server
```bash
python api.py
```
Server จะรันที่ http://localhost:5000

### 5. ทดสอบ API
```bash
curl -X POST http://localhost:5000/analyze \
     -F "video=@วิดีโอแมว.mp4"
```

## ผลลัพธ์ JSON ที่ได้

```json
{
  "personality": {
    "playful": 0.75,
    "lazy": 0.15,
    "shy": 0.10,
    "dominant": "playful",
    "confidence": 0.75
  },
  "appearance": {
    "fur_color_primary": "#D4822A",
    "fur_color_secondary": "#FFFFFF",
    "fur_length": "short",
    "fur_pattern": "solid"
  },
  "frames_analyzed": 10
}
```

## แนวทางพัฒนาต่อ
- เพิ่ม class นิสัยใหม่ เช่น "aggressive", "curious"
- เพิ่ม data มากขึ้นเพื่อความแม่น
- ใช้ YOLO ตรวจจับแมวในภาพให้แม่นขึ้น
- เพิ่ม audio analysis กลับเข้ามา
