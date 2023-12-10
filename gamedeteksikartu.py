
import cv2
import numpy as np
import copy
import ModulKlasifikasiCitraCNN as mCNN
import requests 
import time
import random

# untuk display text
def DrawText(img,sText,pos):
    font        = cv2.FONT_HERSHEY_SIMPLEX
    posf        = pos
    fontScale   = .7
    fontColor   = (255,0,255)
    thickness   = 2
    lineType    = 2
    cv2.putText(img,sText,
        posf,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)
    return copy.deepcopy(img)

# inisialisasi kartu komputer awal 7 kartu
def initCardCom():
    global cardCom
    global labelkelas
    global labelKelasList
    # komputer mengambil 4 kartu acak
    for handCom in range(0,4):
        # iterasi hingga mendapatkan kartu acak yang berbeda
        while True:
            randomNumber = random.randint(0, len(labelKelas) - 1)
            if randomNumber not in cardCom:
                if labelKelas[randomNumber] in labelKelasList:
                    cardCom.append(randomNumber)
                    break
    # kartu yang sudah muncul tidak boleh muncul lagi
    for removeIndex in range(0, len(cardCom)):
        labelKelasList.remove(labelKelas[cardCom[removeIndex]])

# draw kartu untuk komputer
def drawCardCom():
    global labelKelas
    global cardCom
    global labelKelasList
    # iterasi hingga mendapatkan kartu acak yang berbeda
    while True:
        randomNumber = random.randint(0, len(labelKelas) - 1)
        if randomNumber not in cardCom:
            if labelKelas[randomNumber] in labelKelasList:
                break
    return randomNumber

# inisialisasi permainan crazy eight
def initCrazyEight():
    cardDiscard = random.randint(0, 51)
    return cardDiscard

# cardPlayer adalah kartu You dan Com
# parameter menggunakan satu kartu dari tangan player
# cardDiscard adalah kartu yang dibuang di daerah tengah (sebagai kondisi permaian)
def crazyEight (cardPlayerIndex, cardDiscardIndex):
    
    kelas = ["club", "spade", "diamond", "heart"]
    rank = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
    
    kelasDiscard = kelas[cardDiscardIndex // 13]
    rankDiscard = rank[cardDiscardIndex % 13]
    
    kelasPlayer = kelas[cardPlayerIndex // 13]
    rankPlayer = rank[cardPlayerIndex % 13]
    
    if kelasDiscard == kelasPlayer or rankDiscard == rankPlayer or rankPlayer == '8':
        return cardPlayerIndex
    elif rankDiscard == '8' and kelasDiscard != kelasPlayer:
        return cardPlayerIndex
    else:
        return None

# untuk simpan indeks kartu ke listkartu
def initCardYou(cardIndex):  
    # mendefinisikan variable global
    global cardYou
    global labelKelasList
    global labelKelas
    if labelKelas[cardIndex] not in labelKelasList:
        print("Kartu sudah terpakai")
        print('\n')
        # DrawTextCustom(frame, "Kartu Sudah Terpakai", (100,30), 1.2, (255, 0, 0), 4)
    else:
        cardYou.append(cardIndex)
        labelKelasList.remove(labelKelas[cardIndex])

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

# untuk menghitung score kartu You
def cardScoreYou ():
    global cardYou
    # rank = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
    # format urutannya seperti diatas, dibawah nilai setiap kartu
    scoreLabel = [2, 3, 4, 5, 6, 7, 50, 9, 10, 1, 10, 10, 10]
    score = 0
    for index in range(0, len(cardYou)):
        score += scoreLabel[index % 13]

    return score

# untuk menghitung score kartu You
def cardScoreCom ():
    global cardCom
    # rank = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
    # format urutannya seperti diatas, dibawah nilai setiap kartu
    scoreLabel = [2, 3, 4, 5, 6, 7, 50, 9, 10, 1, 10, 10, 10]
    score = 0
    for index in range(0, len(cardCom)):
        score += scoreLabel[index % 13]

    return score

def openWindowInit(url):
    # mendefinisikan variable global
    global cardYou
    global labelKelasList
    global labelKelas
    global model
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
         print(f"Cannot open camera for Input Card")
         exit()

    while True:
         # Capture frame-by-frame
         ret, frameOrigin = cap.read()
         frame = copy.deepcopy(frameOrigin)
         frame = cv2.resize(frame, (900, 600))
         key = cv2.waitKey(5)
        
         # if frame is read correctly ret is True
         if not ret:
            print(f"Can't receive frame for Input Card (stream end?). Exiting ...")
            break

         # # hanya menggunakan region of interest (ROI) yang kiri ((0,0), (293,600))
         # width, height = frame.shape[1], frame.shape[0]
         # left_roi = frame[:, :width//3, :]
         
         imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # imGray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
                                                                                       
         imThres = cv2.adaptiveThreshold(imGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,71,10)
         # cv2.imshow("Citra threshold adaptif", imThres)

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
          # semua rectangle component masuk list
         rectangles = [[], []]
         for i in bigComponent:
             topLeft = values[i,0:2]
             bottomRight = values[i,0:2]+values[i,2:4]
      
             rectangles[0].append(topLeft)
             rectangles[1].append(bottomRight)
             # break
          # semua rectangle digambar
         for i in range(len(rectangles[0])):
             frame = cv2.rectangle(frame, rectangles[0][i], rectangles[1][i], color=(255, 0, 255), thickness=3)
         # cv2.imshow("Mendeteksi kartu", frame)
         
         # buat threshold akurasi prediksi agar meminimalisir prediksi tidak tepat
         thresholdPredict = 0.8
          
          #connected componen dicrop menjadi objek kartu dan diprediksi
         for i in bigComponent:
             topLeft = values[i,0:2]
             bottomRight = values[i,0:2]+values[i,2:4]
              
             cardImage = imThres[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
             
             cardImage = cv2.cvtColor(cardImage, cv2.COLOR_GRAY2BGR)

             # Feed into model
             X = []
             img = cv2.resize(cardImage,(128,128))
             img = np.asarray(img)/255
             img = img.astype('float32')
             X.append(img)
             X = np.array(X)
             X = X.astype('float32')

             # indeks labelKelas dengan hs tertinggi adalah hasil prediksi 
             # Get the prediction scores
             hs = model.predict(X,verbose = 0)
             # n = np.max(np.where(hs== hs.max()))

             # cek apakah skor prediksi maksimum melebihi threshold prediksi
             if np.max(hs) > thresholdPredict:
                 # dapatkan indeks dari skor prediksi maksimum melebihi threshold prediksi
                 n = np.argmax(hs)
                 
                 if labelKelas[n] in labelKelasList:
                     # Put text into image
                     textCoordinate = topLeft + [0, -10]
                     DrawText(frame, f'{labelKelas[n]} {"{:.2f}".format(hs[0,n])}', textCoordinate - [10,0])
                 else:
                     # Put text into image
                     textCoordinate = topLeft + [0, -10]
                     DrawText(frame, 'kartu sudah terpakai', textCoordinate - [10,0])
                     DrawText(frame, f'{labelKelas[n]} {"{:.2f}".format(hs[0,n])}', textCoordinate - [10,20])

         cv2.imshow('Init Card', frame)

        # untuk menghentikan game
         if key == 27 :
             break

        # If space key is pressed, close the current window and break the loop
         if key == 32:
             # kartu yang dipegang You bertambah
             initCardYou(n)
             # You mengambil 4 kartu
             if len(cardYou) == 4:
                 break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyWindow('Init Card')
    
def openWindowDraw(url):
    # mendefinisikan variable global
    global cardYou
    global labelKelasList
    global labelKelas
    global model
    
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
         print(f"Cannot open camera for Input Card")
         exit()

    while True:
         # Capture frame-by-frame
         ret, frameOrigin = cap.read()
         frame = copy.deepcopy(frameOrigin)
         frame = cv2.resize(frame, (900, 600))
         key = cv2.waitKey(5)
        
         # if frame is read correctly ret is True
         if not ret:
            print(f"Can't receive frame for Input Card (stream end?). Exiting ...")
            break

         # # hanya menggunakan region of interest (ROI) yang kiri ((0,0), (293,600))
         # width, height = frame.shape[1], frame.shape[0]
         # left_roi = frame[:, :width//3, :]
         
         imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # imGray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
                                                                                       
         imThres = cv2.adaptiveThreshold(imGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,71,10)
         # cv2.imshow("Citra threshold adaptif", imThres)

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
          # semua rectangle component masuk list
         rectangles = [[], []]
         for i in bigComponent:
             topLeft = values[i,0:2]
             bottomRight = values[i,0:2]+values[i,2:4]
      
             rectangles[0].append(topLeft)
             rectangles[1].append(bottomRight)
             # break
          # semua rectangle digambar
         for i in range(len(rectangles[0])):
             frame = cv2.rectangle(frame, rectangles[0][i], rectangles[1][i], color=(255, 0, 255), thickness=3)
         # cv2.imshow("Mendeteksi kartu", frame)
         
         # buat threshold akurasi prediksi agar meminimalisir prediksi tidak tepat
         thresholdPredict = 0.8
          
          #connected componen dicrop menjadi objek kartu dan diprediksi
         for i in bigComponent:
             topLeft = values[i,0:2]
             bottomRight = values[i,0:2]+values[i,2:4]
              
             cardImage = imThres[topLeft[1]:bottomRight[1],topLeft[0]:bottomRight[0]]
             
             cardImage = cv2.cvtColor(cardImage, cv2.COLOR_GRAY2BGR)

             # Feed into model
             X = []
             img = cv2.resize(cardImage,(128,128))
             img = np.asarray(img)/255
             img = img.astype('float32')
             X.append(img)
             X = np.array(X)
             X = X.astype('float32')

             # indeks labelKelas dengan hs tertinggi adalah hasil prediksi 
             # Get the prediction scores
             hs = model.predict(X,verbose = 0)
             # n = np.max(np.where(hs== hs.max()))

             # cek apakah skor prediksi maksimum melebihi threshold prediksi
             if np.max(hs) > thresholdPredict:
                 # dapatkan indeks dari skor prediksi maksimum melebihi threshold prediksi
                 n = np.argmax(hs)
                 
                 if labelKelas[n] in labelKelasList:
                     # Put text into image
                     textCoordinate = topLeft + [0, -10]
                     DrawText(frame, f'{labelKelas[n]} {"{:.2f}".format(hs[0,n])}', textCoordinate - [10,0])
                 else:
                     # Put text into image
                     textCoordinate = topLeft + [0, -10]
                     DrawText(frame, 'kartu sudah terpakai', textCoordinate - [10,0])
                     DrawText(frame, f'{labelKelas[n]} {"{:.2f}".format(hs[0,n])}', textCoordinate - [10,20])

         cv2.imshow('Draw Card', frame)

        # untuk menghentikan game
         if key == 27 :
             break

        # If space key is pressed, close the current window and break the loop
         if key == 32:
             # kartu yang dipegang You bertambah
             initCardYou(n)
             break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyWindow('Draw Card')
    
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
# buat window game over
def gameOver(siapaMenang):
    global finish
    
    cv2.namedWindow('GAME OVER')
    
    while True:
         frame = np.zeros((300, 400, 3), dtype=np.uint8)
         key = cv2.waitKey(5)
         DrawTextCustom(frame, "GAME OVER", (70,130), 1.5, (255, 255, 255), 5)
         DrawTextCustom(frame, f"{siapaMenang} WIN", (100,180), 1.5, (255, 255, 255), 5)
         cv2.imshow('GAME OVER', frame)

        # untuk menghentikan game
         if key == 27:
             finish = True
             break
    
    cv2.destroyWindow('GAME OVER')
    
# buat game frame
def gameFrame(key):
    global labelKelas
    global labelKelasList
    global turn
    global cardYou
    global cardCom
    global gameIsStarting
    global clickRoiCardYou
    global displayCardYou
    global displayCardCom
    global cardDiscard
    global count
    global countStart
    global textLog
    global textAvailableCard
    global textScoreYou
    global textScoreCom
    global textTurn
    # buat frame hitam 700x1200
    frame = np.zeros((700, 1200, 3), dtype=np.uint8)
    # buat region of interest game board, logging, dan scoring
    roiGame = frame[0:700, 0:900]
    roiLogging = frame[0:350, 900:1200]
    roiScoring = frame[350:700, 900:1200]
    # frame menjadi hijau
    roiGame[:, :] = [0, 255, 0]  # Atur warna ke hijau (R, G, B)
    # frame menjadi merah dan biru
    roiLogging[:, :] = [255, 0, 0]  # Atur warna ke hijau (R, G, B)
    roiScoring[:, :] = [0, 0, 255]  # Atur warna ke hijau (R, G, B)
    # tiap roi dibuat garis kotak
    frame = cv2.rectangle(frame, (0,0), (900,700), color=(0, 0, 0), thickness=5)
    frame = cv2.rectangle(frame, (900,0), (1200,347), color=(0, 0, 0), thickness=5)
    frame = cv2.rectangle(frame, (900,350), (1200,697), color=(0, 0, 0), thickness=5)
    # penamaan setiap region
    DrawTextCustom(frame, "LOG", (920,50), 1.2, (0, 0, 0), 5)
    DrawTextCustom(frame, "Sisa Kartu:", (920,300), 1.2, (0, 0, 0), 4)
    DrawTextCustom(frame, "SCORE", (920,400), 1.2, (0, 0, 0), 5)
    DrawTextCustom(frame, "You:", (920,500), 1.2, (0, 0, 0), 4)
    DrawTextCustom(frame, "Com:", (920,600), 1.2, (0, 0, 0), 4)
    DrawTextCustom(frame, textLog, (920,200), 1.2, (0, 0, 0), 4)
    DrawTextCustom(frame, textTurn, (920,100), 1, (0, 0, 0), 4)
    DrawTextCustom(frame, textAvailableCard, (1140,300), 1.2, (0, 0, 0), 4)
    DrawTextCustom(frame, textScoreYou, (1050,500), 1.2, (0, 0, 0), 4)
    DrawTextCustom(frame, textScoreCom, (1050,600), 1.2, (0, 0, 0), 4)
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
    # display semua kartu yang dipegan You
    for index in range(0, len(cardYou)):
        roiCardYou.append(frame[hMin:hMax, wMin:wMax])
        frame = cv2.rectangle(frame, (wMin,hMin), (wMax - 3,hMax), color=(0, 0, 0), thickness=5)
        # load image
        cardImage = cv2.imread(f"card/{labelKelas[cardYou[index]]}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (100, 100))
        roiCardYou[index][:] = 0  # Clear roi
        roiCardYou[index][:] = cardImageResized
        # print(wMin)
        wMin += 100
        wMax += 100
    # jika You ingin discard kartu dengan klik kartu tersebut, terhubung ke kondisi clickRoiCardYou
    if displayCardYou != None:
        # kartu masuk ROI You
        # load image
        cardImage = cv2.imread(f"card/{labelKelas[displayCardYou]}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (200, 300))
        roiYou[:] = 0  # Clear roi
        roiYou[:] = cardImageResized
    # display cardCom yang mau didiscard
    if displayCardCom != None:
        # kartu masuk ROI You
        # load image
        cardImage = cv2.imread(f"card/{labelKelas[displayCardCom]}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (200, 300))
        roiCom[:] = 0  # Clear roi
        roiCom[:] = cardImageResized
    # jika You ingin discard kartu dengan klik kartu tersebut, terhubung ke kondisi clickRoiCardYou
    if cardDiscard != None:
        # kartu masuk ROI Discard
        # load image
        cardImage = cv2.imread(f"card/{labelKelas[cardDiscard[len(cardDiscard)-1]]}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (200, 300))
        roiDiscard[:] = 0  # Clear roi
        roiDiscard[:] = cardImageResized
    # Giliran Com habis You
    if turn % 2 == 1:
        if count >= 50:
            # cek untuk semua kartu COm ada yang sesuai dengan kartu discard
            for index in range (0, len(cardCom)):
                if crazyEight(cardCom[index], cardDiscard[len(cardDiscard) - 1]) is not None:
                    displayCardCom = cardCom[index]
                    cardDiscard.append(cardCom[index])
                    cardCom.pop(index)
                    turn += 1
                    textLog = 'Com Discard'
                    break
                # jika tidak ada akan draw 1 kartu
                elif crazyEight(cardCom[index], cardDiscard[len(cardDiscard) - 1]) is None:
                    if index == (len(cardCom) - 1):
                        # jika tidak ada kartu yang sesuai, draw satu kartu
                        textLog = 'Com Draw'
                        cardCom.append(drawCardCom())
                        turn += 1
                        break
            count = 0
            countStart = False
    # jika game sudah dimulai dan You draw satu kartu. Setelah muncul draw window harus refresh window game
    if key == 32:
        countStart = True
        wMin = (len(cardYou)-1)*100
        wMax = wMin + 100
        roiCardYou.append(frame[hMin:hMax, wMin:wMax])
        frame = cv2.rectangle(frame, (wMin,hMin), (wMax - 3,hMax), color=(0, 0, 0), thickness=5)
        # load image
        cardImage = cv2.imread(f"card/{labelKelas[cardYou[len(cardYou) - 1]]}.jpg", -1)
        # Resize image agar sesuai dengan region yang diinginkan
        cardImageResized = cv2.resize(cardImage, (100, 100))
        roiCardYou[index][:] = 0  # Clear roi
        roiCardYou[index][:] = cardImageResized
        turn += 1
    # penamaan region
    DrawTextCustom(frame, "Your Card", (10,580), 1.2, (0, 0, 0), 4)
    # ada event mouse click
    if clickRoiCardYou != None:
        if roiCardYou and cardYou:
            countStart = True
            # simpan kartu yang diklik nanti akan di display di roiYou
            displayCardYou = cardYou[clickRoiCardYou]
            if crazyEight(cardYou[clickRoiCardYou], cardDiscard[len(cardDiscard) - 1]) is not None:
                cardDiscard.append(cardYou[clickRoiCardYou])
            else:
                print("DRAW")
            # kartu yang dipegang berkurang
            for index in range(clickRoiCardYou,len(roiCardYou)):
                if index < len(roiCardYou) - 1: 
                    # load image
                    cardImage = cv2.imread(f"card/{labelKelas[cardYou[index + 1]]}.jpg", -1)
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
            turn += 1
    if countStart == True:
        count += 1
    # untuk update nilai text
    textAvailableCard = f'{len(labelKelasList)}'
    textScoreYou = f'{cardScoreYou()}'
    textScoreCom = f'{cardScoreCom()}'
    if turn % 2 == 0:
        textTurn = f'Turn ke-{turn+1} (YOU)'
    else:
        textTurn = f'Turn ke-{turn+1} (COM)'
    # kondisi penyelesaian game
    if not cardCom:
        gameOver('COM')
    elif not cardYou:
        countStart = True
        if count > 10:
            gameOver('YOU')
    elif not labelKelasList:
        countStart = True
        if count > 10:
            if cardScoreYou() <= cardScoreCom():
                gameOver('YOU')
            else:
                gameOver('COM')
    
    return frame

def gameEngine():
    global gameIsStarting
    global clickRoiCardYou
    global url
    global finish
    # Set up the window and set the mouse event callback
    cv2.namedWindow('Game')
    cv2.setMouseCallback('Game', click_event)
    # Main loop
    while True:
        clickRoiCardYou = None
        key = cv2.waitKey(5)
        # Break the loop if the 'Esc' key is pressed
        if key == 27 or finish:
            break
        if key == 32:
            openWindowDraw(url)
        frame = gameFrame(key)
        # Display the frame with regions
        cv2.imshow('Game', frame)

    # Release the OpenCV window
    cv2.destroyAllWindows()
    # key = 2
    # frame = gameFrame(key)
    # cv2.imshow('Game', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
 
    
 
    
######################################## MAIN #######################################
# url = "http://192.168.50.2:8080/video"
# # url pake hotspot
url = "http://192.168.18.190:8080/video"
# url pake WIFI TOWER 2
# url = "http://10.4.67.44:8080/video"

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
finish = False
n = None
clickRoiCardYou = None
displayCardYou = None
displayCardCom = None
turn = 0
count = 0
countStart = False
textLog = '-'
textScoreYou = '-'
textScoreCom = '-'
textAvailableCard = '-'
textTurn = '-'
cardDiscard = []
# turn 0 inisialisasi permainan crazy eight
# cardDiscard = initCrazyEight()
firstDiscardCard = initCrazyEight()
# kartu yang sudah muncul tidak boleh muncul lagi
labelKelasList.pop(firstDiscardCard)
# masuk ke list index Discard card yang sudah muncul secara berurutan
cardDiscard.append(firstDiscardCard)
# isinya indeks absolut kartu
cardCom = []
# isinya index pada labelKelasList, harus diubah ke index pada labelKelas
initCardCom()
# isinya indeks absolut kartu
# cardYou = []
cardYou = [6,19,32,45]
# nanti prediksi semua kartu yang You pegang pakai kamera baru mulai turn
gameIsStarting = False
# load model
model = mCNN.LoadModel("BobotKartu.h5")
timeStart = time.time()


# openWindowInit(url)
gameEngine()