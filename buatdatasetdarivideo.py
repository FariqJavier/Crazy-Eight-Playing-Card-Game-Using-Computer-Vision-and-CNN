import cv2
import os
import datetime
import time
# import ModulKlasifikasiCitraCNN as mCNN
import numpy as np


cardName = [
            "2c","3c","4c","5c","6c","7c","8c","9c","10c","Ac","Jc","Qc","Kc",
            "2s","3s","4s","5s","6s","7s","8s","9s","10s","As","Js","Qs","Ks",
            "2d","3d","4d","5d","6d","7d","8d","9d","10d","Ad","Jd","Qd","Kd",
            "2h","3h","4h","5h","6h","7h","8h","9h","10h","Ah","Jh","Qh","Kh"
]

# index label
cardNameIndex = 0

# penamaan file
def buatNamaFile():
    # ambil keterangan waktu saat ini
    t = datetime.datetime.now()
    # buat nama file dengan format Year-Month-Day-HourMinuteSecondF
    namaFile = t.strftime('%Y-%m-%d-%H%M%S%f')
    return namaFile

# luas dari 4 titik 
def formulaShoelace(points):
    # 'points' adlh input berupa array yg berisikan 4 titik koordinat kartu
    points = np.vstack((points, points[0]))
    area = 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
    return area


def buatDirektori(path):
    ls = []
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)
    for i in range(len(ls)-2,-1,-1):
        sf = ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)

def buatDataSet(sDirektoriData,NoKamera,FrameRate):
    global cardName, cardNameIndex

    # For webcam input:
    cap = cv2.VideoCapture(NoKamera)
    # untuk url hotspot pribadi
    # url = "http://192.168.1.17:8080/video"
    # untuk url wifi B401
    # url = "http://192.168.50.2:8080/video"
    # cap = cv2.VideoCapture(url)
    TimeStart = time.time()

    # limit data kartu yang didapat dlm beberapa detik
    saveTimeLimit = time.time()

    # penandaan ketika kondisi record data
    isSaving = False

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.resize(frame,(600,400))
        
        if cardNameIndex > 51:
            break
        sDirektoriKelas = sDirektoriData+"/"+cardName[cardNameIndex]
        buatDirektori(sDirektoriKelas)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # inisialisasi apakah kartu terdeteksi
        isDetected = False
        
        imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                                                                      
        imThres = cv2.adaptiveThreshold(imGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,71,10)
        cv2.imshow("Citra ", imThres)

        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(imThres, 4, cv2.CV_32S)
        
        # cari connected componen yang sesuai
        # jika ada connected component sesuai kriteria, masuk list bigComponent
        bigComponent = []
        for i in range(totalLabels):
            dimensiComponent = values[i,2:4]
            
            # jika ada komponen dengan 50 <= panjang <= 200 dan 200 <= lebar <= 450
            # termasuk kategori bigComponent
            if (50<dimensiComponent[0]<250 and 200<dimensiComponent[1]<500):
                bigComponent.append(i)
            elif (50<dimensiComponent[1]<250 and 200<dimensiComponent[0]<500):
                bigComponent.append(i)

        # gambar rectangle dari componen tersebut
        for i in bigComponent:
            topLeft = values[i,0:2]
            bottomRight = values[i,0:2]+values[i,2:4]
    
            frame = cv2.rectangle(frame, topLeft, bottomRight, color=(0,0,255), thickness=3)

            break
        cv2.imshow("Mendeteksi adanya kartu", frame)
        
        #connected componen dicrop menjadi objek kartu
        for i in bigComponent:
            topLeft = values[i,0:2]
            bottomRight = values[i,0:2]+values[i,2:4]
            
            cardImage = imThres[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
            
            # cv2.imshow('Crop objek kartu', cardImage)
            break
        
        #mulai record
        TimeNow = time.time()
        if TimeNow-TimeStart>1/FrameRate:
            sfFile = sDirektoriKelas+"/"+buatNamaFile()
            #record teken spasi
            if isSaving and len(bigComponent) > 0:
                cv2.imwrite(sfFile+'.jpg', cardImage)
            TimeStart = TimeNow

        cv2.putText(frame, "Nama Kartu yg direkam:", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"{cardNameIndex+1}. " + cardName[cardNameIndex], (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Tekan spasi untuk mulai record", (0, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        if isSaving:
            cv2.putText(frame, "Record", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            saveTimeLimit = time.time()

        cv2.imshow("Tampilan akhir", frame)
        
        key = cv2.waitKey(5)

        #tekan spasi untuk record
        if key == 32:
            isSaving = not isSaving

        # direcord lima detik
        if time.time() - saveTimeLimit >= 5:
            cardNameIndex += 1
            if cardNameIndex > 51:
                break
            isSaving = False

        if key & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

#direktori dataset
DirektoriDataSet = "dataset"

buatDataSet(DirektoriDataSet, NoKamera=0, FrameRate=20)