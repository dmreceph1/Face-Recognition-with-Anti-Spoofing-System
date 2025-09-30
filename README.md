# Face Recognition with Anti-Spoofing System 🛡️

Bu proje, **yapay zeka destekli yüz tanıma** ve **fake/real yüz ayrımı** yapan bir sistemdir. Gerçek kişiler tespit edildiğinde Firebase veritabanından kullanıcı bilgileri çekilir ve görsel arayüzde görüntülenir. Fake (sahte) yüzler sistem tarafından tespit edilip göz ardı edilir.

---
⚠️ Not: Eğitimli CNN modelim (face_liveness_model.h5) boyutu büyük olduğundan GitHub’a yüklenmemiştir. Bu modeli ayrıca paylaşacağım. Projeyi çalıştırmadan önce, modeli kendi makinenize indirip proje dizinine eklemeniz gerekmektedir.
---
⚠️ Not: Firebase servisi için JSON key dosyasını kendiniz oluşturup proje dizinine eklemeniz gerekmektedir; proje bu dosyayı otomatik olarak kullanacaktır.

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

# 1️⃣ Depoyu klonlayın
git clone <repo-link>
cd <proje-dizini>

# 2️⃣ Gerekli Python paketlerini yükleyin
pip install -r requirements.txt

# 3️⃣ CNN Modelini ekleyin
echo "⚠️ Eğitimli CNN modelim (face_liveness_model.h5) boyutu büyük olduğundan GitHub'a yüklenmemiştir."
echo "Lütfen modeli indirip proje dizinine ekleyin."

# 4️⃣ Firebase JSON key dosyasını ekleyin
echo "⚠️ Firebase servisi için JSON key dosyasını kendiniz oluşturup proje dizinine eklemeniz gerekiyor."
echo "Proje bu dosyayı otomatik olarak kullanacaktır."

# 5️⃣ Projeyi çalıştırın
python app.py
