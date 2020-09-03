from keras.models import Model, load_model
import numpy as np
import os

import argparse
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

model = load_model('HPRmodelv2.h5') # 모델 연결.

#모델의 라벨불러오기
def load_label():
    listfile=[]
    with open("label.txt",mode='r') as l:
        listfile=[i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label
#현재 상황
# - 모델은 잘 읽힘 ( summary를 통해 확인)

# convert작업이 끝나고 나서 그 아웃풋 폴더를 받아와서 읽는다.

def load_data(dirname):
    if dirname[-1] != '/':
        dirname = dirname + '/'
    listfile = os.listdir(dirname)
    print(listfile) # -> 4개 파일 다 나온다.
    X = []
    Y = []
    for file in listfile:
        wordname = file
        print(wordname)
        textlist = os.listdir(dirname + wordname)
        print(textlist)
        for text in textlist:
            textname = dirname + wordname + "/" + text
            #numbers = []
            print(textname)
            with open(textname, mode='r') as t:
                numbers = [float(num) for num in t.read().split()] # 텍스트를 읽어서 숫자단위로 프레임 랜드마크

                #print(len(numbers[0]))
                print("여기까지 가능")
                for i in range(len(numbers), 25200):
                    numbers.extend([0.000])  # 300 frame 고정
            landmark_frame = []
            row = 0
            for i in range(0, 70):  # 총 100프레임으로 고정
                landmark_frame.extend(numbers[row:row + 84])
                row += 84
            landmark_frame = np.array(landmark_frame)
            landmark_frame = landmark_frame.reshape(-1, 84)  # 2차원으로 변환(260*42)
            X.append(np.array(landmark_frame))
            Y.append(textname)
    X = np.array(X)
    Y = np.array(Y)
    print(Y)
    x_train = X
    x_train = np.array(x_train)
    return x_train, Y


output_data_path = "C://Users/pc/slr/output/"
#main함수
def motion_detect():
    output_dir = output_data_path + "Absolute/"
    x_test, Y = load_data(output_dir)
    new_model = tf.keras.models.load_model('HPRmodelv4.h5')
    new_model.summary()

    labels = load_label()

    # 모델 Start
    xhat = x_test
    yhat = new_model.predict(xhat)
    predictions = np.array([np.argmax(pred) for pred in yhat])
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    s = 0
    txtpath = output_data_path + "result.txt"
    with open(txtpath, "w") as f:
        for i in predictions:
            f.write(Y[s])
            f.write(" ")
            f.write(rev_labels[i])
            f.write("\n")
            s += 1

if __name__ == "__main__":
    motion_detect()