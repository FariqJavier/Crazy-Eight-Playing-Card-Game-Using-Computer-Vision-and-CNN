import ModulKlasifikasiCitraCNN as mCNN

DirektoriDataSet = r"C:\Users\fariq\Documents\Coding\Visual Studio Code Coding\Python\PENGOLAHAN CITRA VIDEO\Test Demo Akhir\dataset"

JumlahEpoh = 25

LabelKelas = (
    # "2c","2d","2h","2s","3c","3d","3h","3s","4c","4d","4h","4s",
    # "5c","5d","5h","5s","6c","6d","6h","6s","7c","7d","7h","7s",
    # "8c","8d","8h","8s","9c","9d","9h","9s","10c","10d","10h","10s",
    # "Ac","Ad","Ah","As","Jc","Jd","Jh","Js","Qc","Qd","Qh","Qs",
    # "Kc","Kd","Kh","Ks"
    "2c","3c","4c","5c","6c","7c","8c","9c","10c","Ac","Jc","Qc","Kc",
    "2s","3s","4s","5s","6s","7s","8s","9s","10s","As","Js","Qs","Ks",
    "2d","3d","4d","5d","6d","7d","8d","9d","10d","Ad","Jd","Qd","Kd",
    "2h","3h","4h","5h","6h","7h","8h","9h","10h","Ah","Jh","Qh","Kh"
)

mCNN.TrainingCNN(JumlahEpoh, DirektoriDataSet, LabelKelas,"BobotKartu.h5")
# mCNN.ImageAugmentation(DirektoriDataSet,'2c')
# mCNN.ImageAugmentation(DirektoriDataSet,'2d')
# mCNN.ImageAugmentation(DirektoriDataSet,'2h')
# mCNN.ImageAugmentation(DirektoriDataSet,'2s')