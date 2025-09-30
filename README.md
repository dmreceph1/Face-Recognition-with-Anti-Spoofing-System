# Face Recognition with Anti-Spoofing System 🛡️

Bu proje, **yapay zeka destekli yüz tanıma** ve **fake/real yüz ayrımı** yapan bir sistemdir. Gerçek kişiler tespit edildiğinde Firebase veritabanından kullanıcı bilgileri çekilir ve görsel arayüzde görüntülenir. Fake (sahte) yüzler sistem tarafından tespit edilip göz ardı edilir.

---

## Özellikler

- **CNN tabanlı Fake/Real ayrımı**: Kendi eğitilmiş model ile yüzün sahte mi gerçek mi olduğu tespit edilir.  
- **Yüz tanıma**: `face_recognition` kütüphanesi ile Firebase Storage'daki kullanıcı yüzleri ile eşleştirme yapılır.  
- **Gerçek zamanlı video akışı**: OpenCV kullanılarak canlı kamera görüntüsü üzerinde çalışır.  
- **Firebase entegrasyonu**:  
  - Kullanıcı verileri Realtime Database üzerinden çekilir.  
  - Kullanıcı görselleri Cloud Storage üzerinden alınır.  
- **Arayüz modları**: Farklı UI modları ve kişisel bilgilerin görüntülenmesi.

---

## Kullanılan Teknolojiler

- Python 3.x  
- OpenCV  
- TensorFlow / Keras (CNN modeli)  
- Ultralytics YOLOv8  
- Face Recognition  
- Cvzone (GUI için)  
- Firebase Realtime Database & Cloud Storage

---

## Kurulum ve Kullanım

Projeyi kullanmak için önce GitHub’dan klonlayıp projenin klasörüne girin. Daha sonra Python 3 yüklü olduğundan emin olarak gerekli kütüphaneleri yükleyin:
```bash
pip install opencv-python firebase-admin numpy face_recognition cvzone tensorflow ultralytics
```

Firebase servisi için oluşturduğunuz serviceAccountKey.json dosyasını projeye ekleyin. Kendi eğitilmiş CNN modelinizi (fake/real ayrımı için) ve YOLOv8 modelinizi (best.pt) proje dizinine koyun. Arayüz görselleri için Resources/Modes/ klasörüne mod resimlerinizi, arka plan görselini Resources/background.png dosyasına ekleyin. Her şey hazır olduğunda terminalde projenin ana dosyasının bulunduğu klasörde şu komutu çalıştırarak sistemi başlatın:
```bash
python main.py
```
Sistem açıldığında kamera aktif hale gelecek ve canlı görüntü üzerinde çalışacaktır; model önce yüzün fake mi real mi olduğunu sınıflandırır, real olarak tespit edilen yüzlerde face_recognition ile Firebase Storage’daki kayıtlı yüzlerle eşleştirme yapılarak kullanıcı bilgileri ekranda gösterilecektir. Sahte (fake) yüzler tespit edildiğinde işlem atlanacaktır.
```
