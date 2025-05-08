from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
from PIL import Image
import os
import mysql.connector
from datetime import datetime
import uuid
import re
import unicodedata

print("Loading models...")

# Load YOLO model
try:
    yolo_model = YOLO("best.pt")
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit()

# Load PaddleOCR model
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=True)
    print("✅ PaddleOCR model loaded successfully")
except Exception as e:
    print(f"❌ Error loading PaddleOCR model: {e}")
    exit()

# Load Real-ESRGAN model
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device set to: {device}")
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model_path = 'Real-ESRGAN/weights/RealESRGAN_x4plus.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        exit()
        
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=200,
        tile_pad=10,
        pre_pad=0,
        device=device
    )
    print("✅ Real-ESRGAN model loaded successfully")
    
    # Test the model with a small image
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    enhanced, _ = upsampler.enhance(test_img, outscale=4)
    print("✅ Real-ESRGAN model tested successfully")
    
except Exception as e:
    print(f"❌ Error loading Real-ESRGAN model: {e}")
    exit()

print("\n✅ All models loaded successfully\n")

# Arabic translations for characters (if any)
translations = {}

def connect_to_db():
    """Connect to the database."""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='anpr'  # Updated database name
        )
        if conn.is_connected():
            print("✅ Successfully connected to the database.")
            return conn
        else:
            print("❌ Failed to connect to the database.")
            return None
    except mysql.connector.Error as err:
        print(f"❌ Database connection error: {err}")
        return None

def save_plate_and_vehicle(letters, digits, vehicle_image):
    conn = None
    try:
        conn = connect_to_db()
        if not conn:
            return

        db_operator = conn.cursor()

        # Clean the input text
        letters = clean_text(letters)
        digits = clean_text(digits)
        print(f"DEBUG: Cleaned letters='{letters}', digits='{digits}'")

        # 1. Check if the plate exists
        sql_check_plate = "SELECT plate_id FROM plates WHERE letters = %s AND numbers = %s"
        db_operator.execute(sql_check_plate, (letters, digits))
        plate = db_operator.fetchone()

        if not plate:
            print("❌ Plate not found in database.")
            return

        plate_id = plate[0]

        # 2. Check if vehicle exists for this plate
        sql_check_vehicle = "SELECT vehicle_id FROM vehicles WHERE plate_id = %s"
        db_operator.execute(sql_check_vehicle, (plate_id,))
        vehicle = db_operator.fetchone()

        if not vehicle:
            # Insert new vehicle row (one-time setup)
            image_path = f"images/{uuid.uuid4()}.jpg"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, vehicle_image)

            sql_insert_vehicle = "INSERT INTO vehicles (plate_id, vehicle_image, created_at) VALUES (%s, %s, NOW())"
            db_operator.execute(sql_insert_vehicle, (plate_id, image_path))
            conn.commit()
            vehicle_id = db_operator.lastrowid
            print("🚗 New vehicle created and logged.")
        else:
            vehicle_id = vehicle[0]

        # 3. Check last log entry for this vehicle
        sql_last_log = """
            SELECT log_id, status FROM vehicle_logs 
            WHERE vehicle_id = %s 
            ORDER BY log_id DESC 
            LIMIT 1
        """
        db_operator.execute(sql_last_log, (vehicle_id,))
        last_log = db_operator.fetchone()

        if not last_log or last_log[1] == 'logout':
            # Insert login record
            sql_insert_log = "INSERT INTO vehicle_logs (vehicle_id, check_in, status) VALUES (%s, NOW(), 'login')"
            db_operator.execute(sql_insert_log, (vehicle_id,))
            print("🟢 Vehicle logged IN")
        elif last_log[1] == 'login':
            # Update last log to mark as logout
            sql_update_log = """
                UPDATE vehicle_logs 
                SET check_out = NOW(), status = 'logout' 
                WHERE log_id = %s
            """
            db_operator.execute(sql_update_log, (last_log[0],))
            print("🔴 Vehicle logged OUT")

        conn.commit()

    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            db_operator.close()
            conn.close()


def calculate_image_quality(image):
    """Calculate image quality."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    return contrast, sharpness

def enhance_plate_image(plate_img):
    try:
        contrast, sharpness = calculate_image_quality(plate_img)
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        
        if contrast < 50 or sharpness < 100:
            enhanced, _ = upsampler.enhance(plate_rgb, outscale=4)
            print("✅ Image enhanced using Real-ESRGAN")
        else:
            enhanced = plate_rgb
            print("✅ Image is clear, no enhancement needed")
        
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
    except Exception as e:
        print(f"❌ Error enhancing image: {e}")
        return plate_img

def is_english(text):
    """Check if the text is in English."""
    return all(ord('A') <= ord(c) <= ord('Z') or ord('a') <= ord(c) <= ord('z') for c in text)

def remove_diacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def clean_text(text):
    text = text.replace(" ", "")
    # امسح كل أشكال كلمة مصر حتى لو فيها مسافات أو تشكيل
    text = remove_diacritics(text)
    text = re.sub(r"م\s*ص\s*ر", "", text)
    text = re.sub(r"egypt", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text

def split_and_filter_letters(letters):
    groups = [letters[i:i+3] for i in range(0, len(letters), 3)]
    filtered = []
    for group in groups:
        # لو المجموعة فيها أي حرف من كلمة مصر تجاهلها
        if any(c in group for c in 'مصر'):
            continue
        filtered.append(group)
    return ''.join(filtered)

def detect_and_ocr(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Error: Unable to read image")
            return
            
        height, width = image.shape[:2]
        if height > 1200 or width > 1600:
            scale = min(1200/height, 1600/width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
            print(f"✅ Image resized to {new_width}x{new_height}")
            
        results = yolo_model(image)[0]

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            if class_name == "License Plate":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                margin = 10
                h, w = image.shape[:2]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                plate_img = image[y1:y2, x1:x2]

                cv2.imshow("Original Plate", plate_img)
                cv2.waitKey(1)

                enhanced_plate = enhance_plate_image(plate_img)
                
                cv2.imshow("Enhanced Plate", enhanced_plate)
                cv2.waitKey(1)

                result = ocr.ocr(enhanced_plate, cls=True)

                if result and len(result) > 0 and result[0]:
                    letters = ""
                    digits = ""

                    for line in result[0]:
                        if len(line) >= 2 and line[1]:
                            text = clean_text(line[1][0])
                            
                            # Skip if the text is empty after cleaning
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

                    # Combine, clean, then split again
                    full_plate = letters + digits
                    full_plate_clean = clean_text(full_plate)
                    letters_clean = ''.join([c for c in full_plate_clean if not c.isdigit()])
                    digits_clean = ''.join([c for c in full_plate_clean if c.isdigit()])

                    letters_final = split_and_filter_letters(letters_clean)
                    print(f"DEBUG: filtered letters='{letters_final}', digits='{digits_clean}'")
                    save_plate_and_vehicle(letters_final, digits_clean, image)
                    

                    cv2.imshow("Final Result", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("❌ No text found")
                break
        else:
            print("❌ No license plate detected")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# Test
detect_and_ocr("Images\\img3-.jpg")
