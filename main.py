import cv2
import os
import firebase_admin
from firebase_admin import credentials, storage, db
import numpy as np
import face_recognition
import cvzone
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import math

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    
    "databaseURL" : "https://doorproject-cf766-default-rtdb.firebaseio.com/",
    "storageBucket" : "doorproject-cf766.appspot.com"
    
    })

#firebase referansları
ref = db.reference('Persons') #referans alıyor. Bu referans ile 'Persons' düğümündeki verilere ulaşabileceğiz.
bucket = storage.bucket() #bucket referansı alınıyor. Bu, Firebase Cloud Storage'deki dosyaları yüklemek veya indirmek için kullanılır.

# Tüm kişisel verileri çek ve ID'lere göre sakla
data = ref.get() # 'Persons' düğümündeki tüm verileri çekiyor ve data isimli bir değişkene atıyor. 
user_info = {} #Çekilen veriyi düzenlemek için boş bir sözlük oluşturuyoruz. 
for user_id, user_data in data.items(): #data sözlüğündeki her bir kullanıcı verisini (ID ve verileri) döngüye alır.
    user_info[user_id] = user_data #Her kullanıcı ID'sini (user_id) user_info sözlüğüne anahtar olarak ekleyip, o ID'ye ait kullanıcı verilerini (user_data) saklar.

# Firebase Storage'dan tüm yüz görsellerini indirip encoding yap
encoded_user_faces = {} # Kullanıcı ID'lerini anahtar (key) olarak kullanarak her bir yüzün kodlamasını saklamak için boş bir sözlük oluşturulur.
for user_id in user_info.keys(): # user_info sözlüğündeki her bir kullanıcı ID'si için döngü başlatır.
    blob = bucket.blob(f'images/{user_id}.jpg')
    img_data = blob.download_as_string() #Firebase Storage'daki görsel dosyasını indirip, dosyanın içeriğini img_data adlı değişkende saklar.
    img_array = np.frombuffer(img_data, np.uint8) #img_data içeriğini bir numpy dizisine dönüştürür. Bu dizi, görsel verisinin piksel verisi olarak saklanmasını sağlar.
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    user_face_encoding = face_recognition.face_encodings(img)[0] #face_recognition kütüphanesini kullanarak görseldeki yüzü kodlar. Bu kodlama, yüzün karakteristik özelliklerini temsil eder ve tanıma işlemi için kullanılır.
    encoded_user_faces[user_id] = user_face_encoding #Elde edilen yüz kodlamasını, user_id anahtarıyla encoded_user_faces sözlüğüne ekler.


cap = cv2.VideoCapture(0)
cap.set(3,640) #çerçeve genişliği
cap.set(4, 480) #çerçeve yüksekliği

backgroundImg = cv2.imread("C:/Users/recep/.spyder-py3/door/Resources/background.png")

#model içindeki background resimleri döngüyle eklendi
foldermodepath = "C:/Users/recep/.spyder-py3/door/Resources/Modes" #dosyaların yolunu belirttik
modepathlist = os.listdir(foldermodepath) # yukarıdaki klasörlerin dosyalarını döndürüp listeye ekler
"""print(modepathlist)""" # mode içindeki dosyalar eklenmiş listeye görebiliriz
imgModeList = []

for path in modepathlist:
    imgModeList.append(cv2.imread(os.path.join(foldermodepath,path)))

#print(len(imgModeList))      Modda kullanılan görsellerin adetini verecek yani 4 

modeType = 0
counter = 0
imgPerson = []
#img_resized = cv2.resize(img, (640, 480))
model = YOLO("best.pt")
classNames = ["fake", "real"]

while True:
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)

    backgroundImg[162:162 + 480, 55:55 + 640] = img
    backgroundImg[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence ve sınıf
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = classNames[cls]

            # Fake veya Real çıktısını yazdır
            print(f"Detected: {label}, Confidence: {conf:.2f}")
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            backgroundImg = cvzone.cornerRect(backgroundImg, bbox, rt=0)

            if label == "real":
                # Yüz bölgesini al
                face_img = img[y1:y2, x1:x2]
                rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(rgb_face_img)

                if face_encoding:
                    face_encoding = face_encoding[0]
                    matches = face_recognition.compare_faces(list(encoded_user_faces.values()), face_encoding)
                    face_distances = face_recognition.face_distance(list(encoded_user_faces.values()), face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        user_id = list(encoded_user_faces.keys())[best_match_index]
                        user_data = user_info[user_id]

                        # Kullanıcı bilgileri
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                        backgroundImg = cvzone.cornerRect(backgroundImg, bbox, rt=0)
                        modeType = 1
                        counter = 1 
  
                                                                              
    if counter != 0 :
        
        if counter == 1:
            studentInfo = db.reference(f'Persons/{user_id}').get()
            #print(studentInfo)
            
            
            blob = bucket.get_blob(f'images/{user_id}.jpg')
            array = np.frombuffer(blob.download_as_string(),np.uint8)
            imgPerson = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
            
            id=user_id
            
            ref = db.reference(f'Persons/{user_id}')
              
        if 10 < counter < 20:
            
             modeType = 2 
             
        backgroundImg [44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        if counter <= 10:
            
            (w,h), _ = cv2.getTextSize(studentInfo["name"], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            offset = (414-w)//2
                
            cv2.putText(backgroundImg, str(studentInfo["name"]), (808+offset,445), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (100,100,100),1)   
            
            
            cv2.putText(backgroundImg, str(id), (1006,493), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (100,100,100),1) 
            
            cv2.putText(backgroundImg, str(studentInfo["dob"]), (1025,625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (100,100,100),1)  
            
            imgPerson_resized = cv2.resize(imgPerson, (216, 213))
            
            backgroundImg[175:172+216, 909:909+216] = imgPerson_resized  
            
            cv2.imshow("background", backgroundImg)
            cv2.waitKey(5000)  # 5 saniye duracak
            counter = 20
            
        counter +=1
        
        if counter >= 20:
            counter = 0
            modeType = 0
            imgPerson = []
            studentInfo = []
            backgroundImg [44:44 + 633, 808:808 + 414] = imgModeList[modeType]
              
    #cv2.imshow("frame", img)
    cv2.imshow("background", backgroundImg)
    cv2.waitKey(1)    