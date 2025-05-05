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
            database='anpr_system'  # Updated database name
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
    """Log or save plate and vehicle data into the database"""
    conn = None
    try:
        conn = connect_to_db()
        if not conn:
            return

        # Using a better name for database operation
        db_operator = conn.cursor()  # Changed from db_executor to db_operator

        # Check if the plate already exists
        sql_check = "SELECT plate_id FROM plates WHERE letters = %s AND numbers = %s"
        db_operator.execute(sql_check, (letters[::-1], digits[::-1]))
        existing_plate = db_operator.fetchone()

        if existing_plate:
            plate_id = existing_plate[0]
            print(f"✅ Plate already exists in the database with ID: {plate_id}")
            print("🟢 Vehicle login recorded.")
        else:
            # Save new plate data
            sql_plate = "INSERT INTO plates (letters, numbers) VALUES (%s, %s)"
            db_operator.execute(sql_plate, (letters[::-1], digits[::-1]))
            plate_id = db_operator.lastrowid
            conn.commit()
            print(f"✅ New plate saved with ID: {plate_id}")

        # Save vehicle image
        if vehicle_image is not None:
            image_path = f"images/{uuid.uuid4()}.jpg"
            os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure directory exists
            cv2.imwrite(image_path, vehicle_image)

            detected_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sql_vehicle = "INSERT INTO vehicles (plate_id, vehicle_image, created_at) VALUES (%s, %s, %s)"
            db_operator.execute(sql_vehicle, (plate_id, image_path, detected_at))
            conn.commit()

            print("📸 Vehicle login recorded with image.")

        conn.commit()  # Commit to ensure all changes are saved
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
    finally:
        # Ensure closing of operator and connection
        if db_operator:
            db_operator.close()
        if conn:
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

def clean_text(text):
    """Clean unwanted characters."""
    text = text.replace("مصر", "")
    text = text.strip()
    return text

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
                            

                            if is_english(text):
                                continue
                                 
                            for char in text:
                                if is_english(char):
                                    continue
                                    
                                translated = translations.get(char, char)
                                if translated.isnumeric():
                                    digits += translated
                                elif translated.strip():
                                    letters += translated

                    letters = clean_text(letters)
                    
                    if letters or digits:
                        final_text = letters[::-1] + " " + digits[::-1]
                        final_text = clean_text(final_text)
                        
                        if final_text.strip():
                            print("🔍 License Plate (letters then numbers):", final_text)

                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, final_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

                            save_plate_and_vehicle(letters, digits, image)
                            

                            cv2.imshow("Final Result", image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        else:
                            print("❌ No valid text found after cleaning")
                    else:
                        print("❌ No valid text found")
                else:
                    print("❌ No text found")
                break
        else:
            print("❌ No license plate detected")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# Test
detect_and_ocr("Images\\img11.jpg")
