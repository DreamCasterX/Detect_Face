import cv2

def detectFace(img):
    filename = img.split(".")[0]   # 將Str以.為單位拆解為List, 並取index 0 (不要副檔名)
    img = cv2.imread(img)  # 讀取圖檔
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 透過轉換函式轉為灰階影像
    color = (0, 255, 0)  # 定義框的顏色 (RGB)

    # OpenCV 人臉識別分類器
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # 調用偵測識別人臉函式
    faceRects = face_classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    # 大於 0 則檢測到人臉
    if len(faceRects):
        # 框出每一張人臉
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)   # rectangle是矩形   2是粗度

    # 將結果圖片輸出
    cv2.imwrite(filename + "_face.jpg", img)

detectFace("crowd.jpg")