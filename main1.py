from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
from PIL import Image
import os

print("جاري تحميل النماذج...")

# تحميل موديل YOLO
try:
    yolo_model = YOLO("best.pt")
    print("✅ تم تحميل موديل YOLO بنجاح")
except Exception as e:
    print(f"❌ خطأ في تحميل موديل YOLO: {e}")
    exit()

# تحميل موديل PaddleOCR
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=True)
    print("✅ تم تحميل موديل PaddleOCR بنجاح")
except Exception as e:
    print(f"❌ خطأ في تحميل موديل PaddleOCR: {e}")
    exit()

# تحميل موديل Real-ESRGAN
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ تم تحديد الجهاز: {device}")
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model_path = 'Real-ESRGAN/weights/RealESRGAN_x4plus.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ لم يتم العثور على ملف النموذج في: {model_path}")
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
    print("✅ تم تحميل موديل Real-ESRGAN بنجاح")
    
    # اختبار الموديل على صورة صغيرة
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    enhanced, _ = upsampler.enhance(test_img, outscale=4)
    print("✅ تم اختبار موديل Real-ESRGAN بنجاح")
    
except Exception as e:
    print(f"❌ خطأ في تحميل موديل Real-ESRGAN: {e}")
    exit()

print("\n✅ تم تحميل جميع النماذج بنجاح\n")

# الترجمة العربية للأحرف
translations = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
    '7aah': 'ح', 'Daad': 'ض', 'Een': 'ع', 'Heeh': 'ه', 'Kaaf': 'ك', 'Laam': 'ل', 'Meem': 'م', 'Noon': 'ن', 'Saad': 'ص',
    'Seen': 'س', 'Taa': 'ط', 'Wow': 'و', 'Yeeh': 'ي', 'Zeen': 'ز', 'alef': 'أ', 'baa': 'ب', 'daal': 'د', 'geem': 'ج',
    'aa': 'ع', 'g': 'ج', 's': 'س', 'ss': 'ص', 'b': 'ب', 'd': 'د', 't': 'ط', 'h': 'ه', 'k': 'ق', 'f': 'ف',
    'n': 'ن', 'l': 'ل', 'm': 'م', 'w': 'و', 'y': 'ي', 'r': 'ر',
}

def calculate_image_quality(image):
    """حساب جودة الصورة"""
    # حساب التباين
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    
    # حساب الحدة
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    return contrast, sharpness

def enhance_plate_image(plate_img):
    try:
        # حساب جودة الصورة
        contrast, sharpness = calculate_image_quality(plate_img)
        
        # تحويل OpenCV image إلى RGB
        plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        
        # تطبيق super resolution فقط إذا كانت الصورة منخفضة الجودة
        if contrast < 50 or sharpness < 100:
            enhanced, _ = upsampler.enhance(plate_rgb, outscale=4)
            print("✅ تم تحسين الصورة باستخدام Real-ESRGAN")
        else:
            enhanced = plate_rgb
            print("✅ الصورة واضحة، لم يتم استخدام Real-ESRGAN")
        
        # تحسين التباين والسطوع
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # تحويل الصورة المعالجة مرة تانية لـ BGR
        enhanced_img = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        
        # تطبيق إزالة الضوضاء إذا كانت الصورة منخفضة الجودة
        if contrast < 50 or sharpness < 100:
            enhanced_img = cv2.fastNlMeansDenoisingColored(enhanced_img, None, 10, 10, 7, 21)
        
        return enhanced_img
    except Exception as e:
        print(f"❌ خطأ في تحسين الصورة: {e}")
        return plate_img

def is_english(text):
    """التحقق مما إذا كان النص باللغة الإنجليزية"""
    return all(ord('A') <= ord(c) <= ord('Z') or ord('a') <= ord(c) <= ord('z') for c in text)

def clean_text(text):
    """تنظيف النص من الكلمات غير المرغوب فيها"""
    # إزالة كلمة مصر
    text = text.replace("مصر", "")
    # إزالة المسافات الزائدة
    text = text.strip()
    return text

def detect_and_ocr(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("❌ خطأ: لا يمكن قراءة الصورة")
            return
            
        # تغيير حجم الصورة إذا كانت كبيرة جداً
        height, width = image.shape[:2]
        if height > 1200 or width > 1600:
            scale = min(1200/height, 1600/width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
            print(f"✅ تم تغيير حجم الصورة إلى {new_width}x{new_height}")
            
        results = yolo_model(image)[0]

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            if class_name == "License Plate":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # إضافة هامش حول اللوحة
                margin = 10
                h, w = image.shape[:2]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                plate_img = image[y1:y2, x1:x2]

                # عرض الصورة الأصلية للوحة
                cv2.imshow("اللوحة الأصلية", plate_img)
                cv2.waitKey(1)

                # تحسين الصورة باستخدام Real-ESRGAN
                enhanced_plate = enhance_plate_image(plate_img)
                
                # عرض الصورة بعد التحسين
                cv2.imshow("اللوحة بعد التحسين", enhanced_plate)
                cv2.waitKey(1)

                # تطبيق OCR على الصورة المحسنة مباشرة
                result = ocr.ocr(enhanced_plate, cls=True)

                if result and len(result) > 0 and result[0]:
                    letters = ""
                    digits = ""

                    for line in result[0]:
                        if len(line) >= 2 and line[1]:
                            # تنظيف النص وإزالة كلمة مصر
                            text = clean_text(line[1][0])
                            
                            # تجاهل النصوص الإنجليزية
                            if is_english(text):
                                continue
                                
                            for char in text:
                                # تجاهل الحروف الإنجليزية
                                if is_english(char):
                                    continue
                                    
                                translated = translations.get(char, char)
                                if translated.isnumeric():
                                    digits += translated
                                elif translated.strip():
                                    letters += translated

                    # تنظيف النتيجة النهائية من كلمة مصر مرة أخرى
                    letters = clean_text(letters)
                    
                    if letters or digits:
                        final_text = letters[::-1] + " " + digits[::-1]
                        # تنظيف نهائي
                        final_text = clean_text(final_text)
                        
                        if final_text.strip():  # التأكد من أن النص ليس فارغاً
                            print("🔍 لوحة الترخيص (الحروف ثم الأرقام):", final_text)

                            # رسم المستطيل والنص على الصورة الأصلية
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, final_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

                            # عرض النتيجة النهائية
                            cv2.imshow("النتيجة النهائية", image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        else:
                            print("❌ لم يتم العثور على نص صالح بعد التنظيف")
                    else:
                        print("❌ لم يتم العثور على نص صالح")
                else:
                    print("❌ لم يتم العثور على نص")
                break
        else:
            print("❌ لم يتم العثور على لوحة ترخيص")
            
    except Exception as e:
        print(f"❌ خطأ: {str(e)}")

# تجربة
detect_and_ocr("Images\\pi.jpg")