# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 08:29:06 2023

@author: fariq
"""
# import cv2

# def openWindow(url, window_name):
#     # cap = cv2.VideoCapture(url)
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print(f"Cannot open camera for {window_name}")
#         exit()

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (900, 600))
#         key = cv2.waitKey(5)
        
#         # if frame is read correctly ret is True
#         if not ret:
#             print(f"Can't receive frame for {window_name} (stream end?). Exiting ...")
#             break

#         cv2.imshow(window_name, frame)
#         global testCount
#         testCount += 1

#         # untuk menghentikan game
#         if key == 27 :
#             break

#         # If space key is pressed, close the current window and break the loop
#         if key == 32:
#             cv2.destroyWindow(window_name)
#             return key
#             # break
    
#     # When everything done, release the capture
#     cap.release()

# # Set an initial value for key
# key = 0

# testCount = 0

# # Open the first window
# key=openWindow("http://192.168.50.2:8080/video", 'test1')

# # Open the second window if the space key is not pressed
# if key == 32:
#     openWindow("http://192.168.50.2:8080/video", 'test2')

# # Close all windows
# cv2.destroyAllWindows()


import cv2
import numpy as np
import copy
import ModulKlasifikasiCitraCNN as mCNN
import requests 
import time
import random

# untuk display text modifikasi
# lebih customize dari fungsi DrawText
def DrawTextCustom(img, sText, pos, fontScale, fontColor, thickness):
    font        = cv2.FONT_HERSHEY_SIMPLEX
    posf        = pos
    lineType    = 2
    cv2.putText(img,sText,
        posf,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)
    return copy.deepcopy(img)
    
# Function to handle mouse events
def click_event(event, x, y, flags, param):
    global cardYou
    global clickRoiCardYou
    if event == cv2.EVENT_LBUTTONDOWN:
        # cek apakah koordinat mouse masuk ke roi
        if 600 <= y <= 700 and 0 <= x < 100:
            # menghandle aksi for roiCardYou0
            clickRoiCardYou = 0
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 100 <= x < 200:
            # menghandle aksi for roiCardYou1
            clickRoiCardYou = 1
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 200 <= x < 300:
            # menghandle aksi for roiCardYou2
            clickRoiCardYou = 2
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 300 <= x < 400:
            # menghandle aksi for roiCardYou3
            clickRoiCardYou = 3
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 400 <= x < 500:
            # menghandle aksi for roiCardYou4
            clickRoiCardYou = 4
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 500 <= x < 600:
            # menghandle aksi for roiCardYou5
            clickRoiCardYou = 5
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 600 <= x < 700:
            # menghandle aksi for roiCardYou6
            clickRoiCardYou = 6
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 700 <= x < 800:
            # menghandle aksi for roiCardYou7
            clickRoiCardYou = 7
            print(clickRoiCardYou)
        elif 600 <= y <= 700 and 800 <= x < 900:
            # menghandle aksi for roiCardYou8
            clickRoiCardYou = 8
            print(clickRoiCardYou)
# buat game frame
def gameFrame(key):
    global labelKelas
    global turn
    global cardYou
    global gameIsStarting
    global clickRoiCardYou
    global displayCardYou
    # buat frame hitam 700x1200
    frame = np.zeros((700, 1200, 3), dtype=np.uint8)
    # buat region of interest game board, logging, dan scoring
    roiGame = frame[0:700, 0:900]
    roiLogging = frame[0:350, 900:1200]
    roiScoring = frame[350:700, 900:1200]
    # frame menjadi hijau
    roiGame[:, :] = [0, 255, 0]  # Atur warna ke hijau (R, G, B)
    # frame menjadi putih
    roiLogging[:, :] = [255, 0, 0]  # Atur warna ke hijau (R, G, B)
    roiScoring[:, :] = [0, 0, 255]  # Atur warna ke hijau (R, G, B)
    # tiap roi dibuat garis kotak
    frame = cv2.rectangle(frame, (0,0), (900,700), color=(0, 0, 0), thickness=5)
    frame = cv2.rectangle(frame, (900,0), (1200,347), color=(0, 0, 0), thickness=5)
    frame = cv2.rectangle(frame, (900,350), (1200,697), color=(0, 0, 0), thickness=5)
    #----------------------------------------UNTUK GAME BOARD ROI-------------------------------
    # buat region of interest peletakan gambar kartu
    roiYou = frame[150:450, 50:250]
    roiDiscard = frame[150:450, 350:550]
    roiCom = frame[150:450, 650:850]
    # tiap roi dibuat garis kotak
    frame = cv2.rectangle(frame, (50,150), (250,450), color=(100, 200, 0), thickness=5)
    frame = cv2.rectangle(frame, (350,150), (550,450), color=(100, 200, 255), thickness=5)
    frame = cv2.rectangle(frame, (650,150), (850,450), color=(255, 100, 0), thickness=5)
    # penamaan setiap region
    DrawTextCustom(frame, "You", (60,140), 1.2, (100, 200, 0), 4)
    DrawTextCustom(frame, "Discarded", (360,140), 1.2, (100, 200, 255), 4)
    DrawTextCustom(frame, "Com", (660,140), 1.2, (255, 100, 0), 4)
    # buat region of interest peletakan gambar kartu
    # tiap roi dibuat garis kotak
    roiCardYou = []
    wMin = 0
    wMax = 100
    hMin = 590
    hMax = 690
    for index in range(0, len(cardYou)):
        roiCardYou.append(frame[hMin:hMax, wMin:wMax])
        frame = cv2.rectangle(frame, (wMin,hMin), (wMax - 3,hMax), color=(0, 0, 0), thickness=5)
        # load image
        cardImage = cv2.imread(f"card/{cardYou[index]}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (100, 100))
        roiCardYou[index][:] = 0  # Clear roi
        roiCardYou[index][:] = cardImageResized
        # print(wMin)
        wMin += 100
        wMax += 100
    if displayCardYou != None:
        # kartu masuk ROI You
        # load image
        cardImage = cv2.imread(f"card/{displayCardYou}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (200, 300))
        # roiYou[:] = 0  # Clear roi
        roiYou[:] = cardImageResized
    # jika game sudah dimulai dan You draw satu kartu. Setelah muncul draw window harus refresh window game
    if key == 32:
        wMin = (len(cardYou)-1)*100
        wMax = wMin + 100
        roiCardYou.append(frame[hMin:hMax, wMin:wMax])
        frame = cv2.rectangle(frame, (wMin,hMin), (wMax - 3,hMax), color=(0, 0, 0), thickness=5)
        # load image
        cardImage = cv2.imread(f"card/{cardYou[len(cardYou) - 1]}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (100, 100))
        roiCardYou[index][:] = 0  # Clear roi
        roiCardYou[index][:] = cardImageResized
    # penamaan region
    DrawTextCustom(frame, "Your Card", (10,580), 1.2, (0, 0, 0), 4)
    # ada event mouse click
    if clickRoiCardYou != None:
        if roiCardYou and cardYou:
            displayCardYou = cardYou[clickRoiCardYou]
            # kartu yang dipegang berkurang
            for index in range(clickRoiCardYou,len(roiCardYou)):
                if index < len(roiCardYou) - 1: 
                    # load image
                    cardImage = cv2.imread(f"card/{cardYou[index + 1]}.jpg", -1)
                    # Resize image agar sesuai dengan region yang diinginkan
                    cardImageResized = cv2.resize(cardImage, (100, 100))
                    roiCardYou[index][:] = 0  # Clear roi
                    roiCardYou[index][:] = cardImageResized
                    cardYou[index] = cardYou[index+1]
                else:
                    wMin = (len(cardYou)-1)*100
                    wMax = wMin + 100
                    roiCardYou[index][:] = [0, 255, 0]   # green roi
                    # ngilangin rectangle dengan timpa warna yang sama
                    frame = cv2.rectangle(frame, (wMin,hMin), (wMax,hMax), color=(0, 255, 0), thickness=5)
                    roiCardYou.pop(index)
                    cardYou.pop(index)

    return frame

def gameEngine():
    global gameIsStarting
    global clickRoiCardYou
    # Set up the window and set the mouse event callback
    cv2.namedWindow('Game')
    cv2.setMouseCallback('Game', click_event)
    # Main loop
    while True:
        clickRoiCardYou = None
        key = cv2.waitKey(5)
        frame = gameFrame(key)
        # Display the frame with regions
        cv2.imshow('Game', frame)

        # Break the loop if the 'Esc' key is pressed
        if key == 27:
            break
    # Release the OpenCV window
    cv2.destroyAllWindows()
    # key = 2
    # frame = gameFrame(key)
    # cv2.imshow('Game', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
 
# sebagai referensi absolut nilai kartu
labelKelas = (
    "2c","3c","4c","5c","6c","7c","8c","9c","10c","Ac","Jc","Qc","Kc",
    "2s","3s","4s","5s","6s","7s","8s","9s","10s","As","Js","Qs","Ks",
    "2d","3d","4d","5d","6d","7d","8d","9d","10d","Ad","Jd","Qd","Kd",
    "2h","3h","4h","5h","6h","7h","8h","9h","10h","Ah","Jh","Qh","Kh"
)
# diubah jadi list kartu yang belum muncul
# jika ada kartu didraw dari deck, list kartu berkurang
labelKelasList = list(labelKelas)

test = False
counter = 0
breaking = False
n = None
turn = 0
textCom = '-'
textYou = '-'
textSystem = '-'
# cardDiscard = []
# # turn 0 inisialisasi permainan crazy eight
# # cardDiscard = initCrazyEight()
# firstDiscardCard = initCrazyEight()
# # kartu yang sudah muncul tidak boleh muncul lagi
# labelKelasList.pop(firstDiscardCard)
# # masuk ke list index Discard card yang sudah muncul secara berurutan
# cardDiscard.append(firstDiscardCard)
# # isinya indeks absolut kartu
# cardCom = []
# # isinya index pada labelKelasList, harus diubah ke index pada labelKelas
# indeksTemp = initCardCom(labelKelasList)
# # index pada labelkelaslist diubah menjadi index pada labelkelas
# for indeks in range(0, len(indeksTemp)):
#     if labelKelasList[indeksTemp[indeks]] in labelKelas:
#        indexConverted  = labelKelas.index(labelKelasList[indeksTemp[indeks]])
#        cardCom.append(indexConverted)
# # kartu yang sudah muncul tidak boleh muncul lagi
# for removeIndex in range(0, len(cardCom)):
#     labelKelasList.remove(labelKelas[cardCom[removeIndex]])
# # isinya indeks absolut kartu
cardYou = ['2c', '3c', '4c', '5c']
clickRoiCardYou = None
displayCardYou = None
# nanti prediksi semua kartu yang You pegang pakai kamera baru mulai turn
gameIsStarting = False
gameEngine()

# import numpy as np
# import cv2
# # Buat frame hitam 1200x700
# frame = np.zeros((700, 1200, 3), dtype=np.uint8)

# # Buat region of interest game board, logging, dan scoring
# # roiGame = frame[0:900, 0:700]
# # roiLogging = frame[900:1000, 0:700]
# # roiScoring = frame[1000:1100, 0:700]
# roiGame = frame[0:700, 0:900]
# roiLogging = frame[0:700, 900:1000]
# roiScoring = frame[0:700, 1000:1100]

# # Set warna hijau pada bagian game board
# roiGame[:, :] = [0, 255, 0]  # Atur warna ke hijau (R, G, B)

# # Tampilkan frame
# cv2.imshow('Frame', frame)
# cv2.waitKey(0)