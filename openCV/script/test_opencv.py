%matplotlib inline
import numpy as np
import cv2, matplotlib
import matplotlib.pyplot as plt
print(cv2.__version__) # cv2 install check

def read_img(name_i):
    img = cv2.imread(name_i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_face(img_name, cascade_path):
    #画像入力
    img = read_img(img_name)
    width = img.shape[1]
    height = img.shape[0]
    img = cv2.resize(img , (width , height))

    #入力画像をグレー画像に変換
    gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #顔認識実行。特徴量のpathを引数から入力。
    cascade = cv2.CascadeClassifier(cascade_path)

    #minNeighbors=20, minSize=(30, 30)→検出枠が近すぎるのと、小さすぎるのは間引く。
    facerect = cascade.detectMultiScale(gry, scaleFactor=1.05, minNeighbors=20, minSize=(30, 30))

    dst_img = []
    if len(facerect) > 0:
        color = (255, 0, 0)
        for rect in facerect:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            #検出した顔を囲む矩形を作成し、１枚の画像ずつ追加。
            add_image = cv2.resize(img[y : y+h , x : x + w] , (64 , 64))
            dst_img.append(add_image)

            #画像に枠を枠を書く
            cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
        # 画像の表示
        plt.imshow(img)

    return dst_img

# 画像の読み込み
img = read_img('data/train_1/train_0.jpg')
img = read_img('lenna.png')
# 画像の表示
plt.imshow(img)

cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
res_images = detect_face('data/train_1/train_0.jpg' ,  cascade_path)
plt.imshow(res_images)
