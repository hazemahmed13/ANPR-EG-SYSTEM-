import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

if __name__ == "__main__":
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus.pth"
    model_path = os.path.join("weights", "RealESRGAN_x4plus.pth")
    
    print("جاري تحميل نموذج RealESRGAN...")
    download_file(model_url, model_path)
    print("تم التحميل بنجاح!") 