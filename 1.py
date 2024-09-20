from pathlib import Path
import cv2
import numpy as np
import time
import tkinter as tk # sudo apt install python3-tk

ESC = 27
# 畫面數量計數
n = 1
# 存檔檔名用
index = 0
# 人臉取樣總數
total = 10

names = ['People1', 'People2', 'People3']

def saveImage(face_image, index, name):
    folder = f'images/{name}'
    Path(folder).mkdir(parents=True, exist_ok=True)

    filename = f'{folder}/{index:03d}.pgm'
    cv2.imwrite(filename, face_image)
    # print(f'{name} 取樣第 {index} 張: {filename}')
    displayText(f'{name} 取樣第 {index} 張: {filename}')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(2)

def rectangle_training():
#===============================================================================
 # 框圖及訓練
#===============================================================================
    for name in names:
        n = 1
        index = 0
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)

        while n > 0:
            ret, frame = cap.read()
            # frame = cv2.resize(frame, (600, 336))
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #### 在while內
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(
                    frame,
                    (x, y), (x + w, y + h),
                    (0, 255, 0), 3
                )
                if n % 5 == 0:
                    face_img = gray[y: y + h, x: x + w]
                    face_img = cv2.resize(face_img, (400, 400))
                    saveImage(face_img, index, name)
                    index += 1
                    if index >= total:
                        print('get training data done')
                        n = -1
                        break
                n += 1

        #### 在while內
            cv2.imshow('video', frame)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()


    # 訓練
    images = []
    labels = []
    for num, name in enumerate(names):
        print(num)
        for index in range(total):
            filename = f'images/{name}/{index:03d}.pgm'
            print('read ' + filename)
            img = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(num)    # 第一張人臉的標籤為0

    print('training...')
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(images), np.asarray(labels))
    model.save('faces.data')
    print('training done')


model = cv2.face.LBPHFaceRecognizer_create()
model.read('faces.data')
print('load training data done')


def facial():
#===============================================================================
 # 辨識
#===============================================================================
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 336))
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #### 在while內
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(
                frame,
                (x, y), (x + w, y + h),
                (0, 255, 0), 3
            )
            face_img = gray[y: y + h, x: x + w]
            face_img = cv2.resize(face_img, (400, 400))

            val = model.predict(face_img)
            # print('label:{}, conf:{:.1f}'.format(val[0], val[1]))
            # print(names[val[0]])
            if val[1] < 50:
                displayText(names[val[0]])
                cv2.putText(
                    frame, names[val[0]], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3
                )
            else:
                displayText('')


    #### 在while內
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

def closeWindow():
#===============================================================================
 # 關閉視窗
#===============================================================================
    window.destroy()

def displayText(text):
#===============================================================================
 # 更改顯示文字
#===============================================================================
    label.config(text=text, font=("Arial", 20) )
    window.update_idletasks()


window = tk.Tk()
window.title('人臉視覺辨識')
window.geometry('600x400')
window.resizable(False, False)

btn_train = tk.Button(window, text="執行框圖及訓練", command=rectangle_training)
btn_train.pack(pady=10)

btn_recognize = tk.Button(window, text="辨識", command=facial)
btn_recognize.pack(pady=10)

btn_exit = tk.Button(window, text="程式結束", command=closeWindow)
btn_exit.pack(pady=10)

# 文字顯示標籤
label = tk.Label(window, text="")
label.pack(pady=20)


window.mainloop()

# while True:
#     print('1. 執行框圖及訓練')
#     print('2. 辨識')
#     print('3. 結束程式')
#     choice = input("請輸入選項: ")
#
#     if choice == '1':
#         rectangle_training()
#     elif choice == '2':
#         facial()
#     elif choice == '3':
#         print("程式結束")
#         break
#     else:
#         print("無效的輸入，請重新輸入")

