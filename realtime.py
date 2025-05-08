import cv2
import numpy as np
import os
import uuid
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
import torch
import mysql.connector
import re

# تحميل النماذج
print("جاري تحميل النماذج...")

try:
    yolo_model = YOLO("best.pt")
    print("✅ YOLO Model Loaded")
except Exception as e:
    print(f"❌ YOLO Load Error: {e}")
    exit()

try:
    ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=True)
    print("✅ PaddleOCR Loaded")
except Exception as e:
    print(f"❌ OCR Load Error: {e}")
    exit()

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model_path = 'Real-ESRGAN/weights/RealESRGAN_x4plus.pth'
    upsampler = RealESRGANer(scale=4, model_path=model_path, model=model,
                             tile=200, tile_pad=10, pre_pad=0, device=device)
    print("✅ Real-ESRGAN Loaded")
except Exception as e:
    print(f"❌ ESRGAN Load Error: {e}")
    exit()

# ترجمة الحروف
translations = {
    # أضف الترجمة لو عندك مثلاً 'ا': 'A'
}

def connect_to_db():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='anpr_system'
        )
        return conn
    except mysql.connector.Error as err:
        print(f"❌ DB Error: {err}")
        return None

def split_and_filter_letters(letters):
    groups = [letters[i:i+3] for i in range(0, len(letters), 3)]
    filtered = []
    for group in groups:
        # لو المجموعة فيها أي حرف من كلمة مصر تجاهلها
        if any(c in group for c in 'مصر'):
            continue
        filtered.append(group)
    return ''.join(filtered)

def clean_text(text):
    text = text.replace(" ", "")
    # امسح كل أشكال كلمة مصر حتى لو فيها مسافات أو تشكيل
    text = re.sub(r"م\s*ص\s*ر", "", text)
    text = re.sub(r"egypt", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text

def plate_exists(letters, numbers):
    conn = connect_to_db()
    if not conn:
        return False
    db_handler = conn.cursor()
    query = "SELECT id FROM plates WHERE letters = %s AND numbers = %s"
    db_handler.execute(query, (letters, numbers))  # Removed the [::-1] here
    result = db_handler.fetchone()
    plate_id = result[0] if result else None
    db_handler.close()
    conn.close()
    return plate_id

def log_vehicle_entry(plate_id, image):
    conn = connect_to_db()
    if not conn:
        return
    db_handler = conn.cursor()
    image_path = f"images/{uuid.uuid4()}.jpg"
    cv2.imwrite(image_path, image)
    detected_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sql = "INSERT INTO vehicles (plate_id, image_path, detected_at) VALUES (%s, %s, %s)"
    db_handler.execute(sql, (plate_id, image_path, detected_at))
    conn.commit()
    db_handler.close()
    conn.close()
    print(f"✅ Vehicle entry logged for plate ID: {plate_id}")

def save_new_plate(letters, numbers, image):
    conn = connect_to_db()
    if not conn:
        return
    db_handler = conn.cursor()
    sql = "INSERT INTO plates (letters, numbers) VALUES (%s, %s)"
    db_handler.execute(sql, (letters, numbers))  # Removed the [::-1] here
    plate_id = db_handler.lastrowid
    conn.commit()
    db_handler.close()
    conn.close()
    print(f"✅ New plate saved with ID: {plate_id}")
    log_vehicle_entry(plate_id, image)

def calculate_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    sharpness = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    return contrast, sharpness

def enhance_plate_image(plate_img):
    contrast, sharpness = calculate_image_quality(plate_img)
    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    if contrast < 50 or sharpness < 100:
        enhanced, _ = upsampler.enhance(plate_rgb, outscale=4)
    else:
        enhanced = plate_rgb
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    enhanced_img = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    if contrast < 50 or sharpness < 100:
        enhanced_img = cv2.fastNlMeansDenoisingColored(enhanced_img, None, 10, 10, 7, 21)
    return enhanced_img

def is_english(text):
    return all(ord('A') <= ord(c) <= ord('Z') or ord('a') <= ord(c) <= ord('z') for c in text)

# ✅ بدء الكاميرا الخارجية
cap = cv2.VideoCapture(1)  # كاميرا خارجية (غير مدمجة)

print("✅ بدأ التشغيل من الكاميرا الخارجية...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ فشل في قراءة الفريم من الكاميرا")
        break

    results = yolo_model(frame)[0]
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = yolo_model.names[class_id]
        if class_name == "License Plate":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            enhanced = enhance_plate_image(plate_img)
            ocr_result = ocr.ocr(enhanced, cls=True)

            if ocr_result and ocr_result[0]:
                letters = ""
                digits = ""

                # First pass: collect all characters
                for line in ocr_result[0]:
                    if len(line) >= 2 and line[1]:
                        text = clean_text(line[1][0])
                        if not text:
                            continue
                        
                        for char in text:
                            if is_english(char):
                                continue
                            translated = translations.get(char, char)
                            if translated.isnumeric():
                                digits += translated
                            elif translated.strip():
                                letters += translated

                # Process letters and digits separately
                letters_clean = ''.join([c for c in letters if not c.isdigit()])
                digits_clean = ''.join([c for c in digits if c.isdigit()])
                
                # Only reverse letters, keep digits as is
                letters_final = split_and_filter_letters(letters_clean)[::-1]
                digits_final = digits_clean[::-1]  # Reverse digits for display

                if letters_final or digits_final:
                    final_text = letters_final + " " + digits_final
                    print(f"🔍 Detected Plate: {final_text}")

                    plate_id = plate_exists(letters_final, digits_clean)  # Use non-reversed digits for DB
                    if plate_id:
                        print("📌 اللوحة موجودة بالفعل، تسجيل دخول فقط")
                        log_vehicle_entry(plate_id, frame)
                    else:
                        print("📌 لوحة جديدة، يتم الحفظ...")
                        save_new_plate(letters_final, digits_clean, frame)  # Use non-reversed digits for DB

                    # عرض النتيجة على الفريم
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, final_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 3, cv2.LINE_AA)
            break

    cv2.imshow("Real-Time License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("✅ تم إيقاف التشغيل")
        break

cap.release()
cv2.destroyAllWindows()
