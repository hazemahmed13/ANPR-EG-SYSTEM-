from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re

# Initialize models
yolo_model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=True)

# Arabic character translations
translations = {
    # ----------------- Digits -----------------
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',

    'digit_0': '٠', 'digit_1': '١', 'digit_2': '٢', 'digit_3': '٣', 'digit_4': '٤',
    'digit_5': '٥', 'digit_6': '٦', 'digit_7': '٧', 'digit_8': '٨', 'digit_9': '٩',

    # ----------- English Transliterations -----------
    '7aah': 'ح', 'Daad': 'ض', 'Een': 'ع', 'Heeh': 'ه',
    'Kaaf': 'ك', 'Laam': 'ل', 'Meem': 'م', 'Noon': 'ن',
    'Saad': 'ص', 'Seen': 'س', 'Taa': 'ط', 'Wow': 'و',
    'Yeeh': 'ي', 'Zeen': 'ز', 'alef': 'أ', 'baa': 'ب',
    'daal': 'د', 'geem': 'ج',

    # ----------- Short Transliteration Variants -----------
    'aa': 'ع', 'g': 'ج', 's': 'س', 'ss': 'ص', 'b': 'ب',
    'd': 'د', 't': 'ط', 'h': 'ه', 'k': 'ق', 'f': 'ف',
    'n': 'ن', 'l': 'ل', 'm': 'م', 'w': 'و', 'y': 'ي', 'r': 'ر',

    # ------------- From Letter Dataset -------------
    'letter_ain': 'ع', 'letter_alef': 'أ', 'letter_baa': 'ب', 'letter_dal': 'د',
    'letter_faa': 'ف', 'letter_haa': 'هـ', 'letter_jeem': 'ج', 'letter_kaf': 'ك',
    'letter_lam': 'ل', 'letter_meem': 'م', 'letter_noon': 'ن', 'letter_qaf': 'ق',
    'letter_raa': 'ر', 'letter_saad': 'ص', 'letter_seen': 'س', 'letter_taa2': 'ط',
    'letter_waaw': 'و', 'letter_yaa': 'ي',
}


def preprocess_image(image):
    """Preprocess the image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def is_arabic_text(text):
    """Check if text contains Arabic characters"""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def is_english_text(text):
    """Check if text contains English characters"""
    english_pattern = re.compile(r'[a-zA-Z]+')
    return bool(english_pattern.search(text))

def is_upper_part_text(text):
    """Check if text is likely to be from the upper part of the plate"""
    # Remove spaces and check for مصر variations
    text_no_spaces = text.replace(" ", "")
    if "مصر" in text_no_spaces:
        return True
        
    # Check for spaced out مصر (م ص ر)
    if all(char in text for char in ["م", "ص", "ر"]):
        return True
        
    # Check for EGYPT variations
    if "EGYPT" in text.upper():
        return True
        
    return False

def process_ocr_result(result):
    """Process OCR results to extract letters and digits"""
    letters = ""
    digits = ""
    
    if not result:
        return None
    
    # Sort lines by vertical position (y-coordinate)
    sorted_lines = sorted(result[0], key=lambda x: x[0][0][1])  # Sort by y-coordinate
    
    for line in sorted_lines:
        text = line[1][0].strip()
        
        # Skip English text
        if is_english_text(text):
            continue
            
        # Skip if it's likely to be the upper part text
        if is_upper_part_text(text):
            continue
            
        # Process only Arabic text and numbers
        for char in text:
            translated = translations.get(char, char)
            if translated.isnumeric():
                digits += translated
            elif translated.strip() and is_arabic_text(translated):
                letters += translated
    
    if not (letters or digits):
        return None
        
    return letters[::-1] + " " + digits[::-1]

def detect_and_ocr(image_path):
    """Main function to detect and recognize license plates"""
    try:
        # Read and validate image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Error: Could not read image")
            return
            
        # Detect license plate
        results = yolo_model(image)[0]
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            if class_name == "License Plate":
                # Extract plate region
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = image[y1:y2, x1:x2]
                
                # Preprocess plate image
                processed_plate = preprocess_image(plate_img)
                
                # Perform OCR
                result = ocr.ocr(processed_plate, cls=True)
                
                # Process OCR results
                final_text = process_ocr_result(result)
                
                if final_text:
                    print("🔍 لوحة الترخيص (الحروف ثم الأرقام):", final_text)
                    
                    # Draw results
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, final_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Show results
                    cv2.imshow("الصورة المعالجة", processed_plate)
                    cv2.imshow("الصورة الكاملة مع النتيجة", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("❌ لم يتم العثور على نص صالح")
                break
        else:
            print("❌ لم يتم العثور على لوحة ترخيص")
            
    except Exception as e:
        print(f"❌ خطأ: {str(e)}")

# Test the function
detect_and_ocr("Images\\img36.jpg")