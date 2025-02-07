from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import magic  # مكتبة لفحص نوع الملف الحقيقي
from fastapi import Depends
from fastapi import HTTPException
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

# ✅ تحميل الموديل المدرب مسبقًا
model = YOLO("yolov8n.pt")  # تأكد من أن لديك هذا الموديل

# ✅ إعداد اتصال MySQL
DATABASE_URL = "mysql+pymysql://anfal:yourpassword@localhost/drone_api"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ إنشاء جدول لحفظ البيانات
class Detection(Base):
    tablename = "detections"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(255))
    confidence = Column(Float)
    image_path = Column(String(255))

Base.metadata.create_all(bind=engine)

# ✅ إنشاء تطبيق FastAPI
app = FastAPI()

# ✅ تقييد الاستضافة لمنع الطلبات غير المصرح بها
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com", "api.yourdomain.com", "localhost"])

# ✅ تفعيل ضغط GZIP لتقليل حجم البيانات المرسلة
app.add_middleware(GZipMiddleware)

# ✅   تفعيل CORS للسماح بطلبات من الويب والتطبيقات
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://theecotrack.com"],  # يمكن استبدال "*" بقائمة محددة من النطاقات
    allow_credentials=True,
    allow_methods=["Post"],  # السماح بجميع أنواع الطلبات (GET, POST, PUT, DELETE)
    allow_headers=["Authorization", "Content-Type"],
)

# ✅ إعداد Rate Limiting لمنع الطلبات الزائدة
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ✅ دالة لإنشاء جلسة اتصال بقاعدة البيانات
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ✅ الصفحة الرئيسية لاختبار تشغيل API
@app.get("/")
def home():
    return {"message": "✅ YOLOv8 API is Running!"}
# ✅ السماح فقط بملفات الصور (تنظيف المدخلات)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 5

def validate_file(file: UploadFile):
    # التحقق من الامتداد
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="❌ الملف غير مدعوم. يُسمح فقط بملفات JPG, JPEG, PNG.")

    # التحقق من حجم الملف
    file.file.seek(0, os.SEEK_END)  # الانتقال إلى نهاية الملف
    file_size = file.file.tell() / (1024 * 1024)  # التحويل إلى ميغابايت
    file.file.seek(0)  # إعادة المؤشر إلى البداية
    if file_size > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"❌ الحد الأقصى لحجم الملف {MAX_FILE_SIZE_MB}MB فقط.")

    # التحقق من نوع الملف الحقيقي
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(file.file.read(2048))
    file.file.seek(0)
    if not file_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="❌ الملف ليس صورة صحيحة.")

    return file

# ✅ API لتحليل الصور وحفظ النتائج في MySQL مع تحسينات الأمان
@app.post("/detect/")
@limiter.limit("5/minute")  # السماح بـ 5 طلبات فقط في الدقيقة لكل مستخدم
async def detect_objects(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        file = validate_file(file)

        # ✅ قراءة الصورة وتحويلها إلى NumPy Array
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ✅ تشغيل YOLOv8 على الصورة
        results = model(image)

        # ✅ حفظ الصورة في السيرفر (تأمين المسار لمنع هجمات الـ Path Traversal)
        sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in ("_", "."))
        image_path = os.path.join("uploads", sanitized_filename)
        os.makedirs("uploads", exist_ok=True)  # إنشاء المجلد إذا لم يكن موجودًا
        cv2.imwrite(image_path, image)

        # ✅ استخراج البيانات من نتائج الكشف وحفظها في قاعدة البيانات
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # إحداثيات الصندوق
                conf = float(box.conf[0])  # نسبة الثقة
                label = result.names[int(box.cls[0])]  # اسم الكائن المكتشف

                detection = Detection(label=label, confidence=conf, image_path=image_path)
                db.add(detection)
                db.commit()

                detections.append({
                    "label": label, 
                    "confidence": round(conf, 2), 
                    "bbox": [x1, y1, x2, y2],
                    "image_path": image_path
                })

        return {"status": "success", "detections": detections}

    except HTTPException as e:
        raise e  # إعادة الخطأ بدون إظهار معلومات حساسة
    except Exception as e:
        return {"error": "❌ حدث خطأ داخلي، الرجاء المحاولة لاحقًا."}

# ✅ تشغيل السيرفر
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)