from ultralytics import YOLO
import cv2

def run_prediction():
    # 1. Eğitimden çıkan EN İYİ modeli yükle
    # Yol: runs/detect/car_damage_v8/weights/best.pt
    model = YOLO("runs/detect/car_damage_v8/weights/best.pt")

    # 2. Test etmek istediğin resmin yolunu ver
    # Örnek: 'data/test/images/ornek_araba.jpg'
    source_img = "data/test/images" # Klasör verirsen hepsini tarar

    # 3. Tahmin yap
    results = model.predict(
        source=source_img,
        conf=0.25,        # Güven eşiği (%25 ve üzeri hasarları göster)
        save=True,        # Sonuçları klasöre kaydet
        device=0          # RTX 4050 GPU kullan
    )

    print(" Tahmin tamamlandı! Sonuçlar 'runs/detect/predict' klasörüne kaydedildi.")

if __name__ == "__main__":
    run_prediction()