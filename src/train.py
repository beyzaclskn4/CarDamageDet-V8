from ultralytics import YOLO

def run_train():
    # 1. Hazır YOLOv8 nano modelini çek (Temel atmak için)
    model = YOLO("yolov8n.pt")

    # 2. Eğitimi başlat
    model.train(
        data="data/data.yaml", # Ayar dosyamız
        epochs=50,             # Kaç tur döneceği 
        imgsz=640,             # Resim boyutu
        batch=8,              # Aynı anda işlenecek resim sayısı
        name="car_damage_v8",  # Çıktı klasörünün adı
        device=0 ,# Ekran kartın varsa 0
        workers=0,
        name="hasar_tespit_ilk_deneme"
    )

if __name__ == "__main__":
    run_train()