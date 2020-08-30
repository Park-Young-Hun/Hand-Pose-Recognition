from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from keras.models import Sequential
from keras import layers
import os
import sys
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import random
from keras import optimizers
from keras.layers import SimpleRNN, Dense
from keras.layers import Bidirectional
import tensorflow as tf
from numpy import argmax
import argparse

#현재 상황
# - 모델은 잘 읽힘 ( summary를 통해 확인)
# -


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
            Y.append(wordname)
    X = np.array(X)
    Y = np.array(Y)
    print(Y)
    x_train = X
    x_train = np.array(x_train)
    return x_train, Y


# prediction
def load_label():
    listfile = []
    with open("label.txt", mode='r') as l:
        listfile = [i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label


def main(input_data_path, output_data_path):
    comp = 'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://python38//python.exe" mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu'
    # 명령어 컴파일
    cmd = 'start bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt'
    # 미디어 파이프 명령어 저장
    listfile = os.listdir(input_data_path)
    if not (os.path.isdir(output_data_path + "Relative/")):
        os.mkdir(output_data_path + "Relative/")
    if not (os.path.isdir(output_data_path + "Absolute/")):
        os.mkdir(output_data_path + "Absolute/")
    for file in listfile:
        # 해당 디렉토리의 하위 디렉토리 폴더명을 찾음
        if not (os.path.isdir(input_data_path + file)):  # ignore .DS_Store
            continue
        word = file + "/"
        fullfilename = os.listdir(input_data_path + word)
        # 하위디렉토리의 모든 비디오들의 이름을 저장
        if not (os.path.isdir(output_data_path + "_" + word)):
            os.mkdir(output_data_path + "_" + word)
        if not (os.path.isdir(output_data_path + "Relative/" + word)):
            os.mkdir(output_data_path + "Relative/" + word)
        if not (os.path.isdir(output_data_path + "Absolute/" + word)):
            os.mkdir(output_data_path + "Absolute/" + word)
        os.system(comp)
        outputfilelist = os.listdir(output_data_path + '_' + word)
        # 동영상 편집.
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list:
                continue
            inputfilen = '   --input_video_path=' + input_data_path + word + mp4list
            outputfilen = '   --output_video_path=' + output_data_path + '_' + word + mp4list
            cmdret = cmd + inputfilen + outputfilen
            os.system(cmdret)
    # mediapipe동작 작동 종료:
    output_dir = output_data_path+ "Absolute/"
    x_test, Y = load_data(output_dir)
    new_model = tf.keras.models.load_model('HPRmodel.h5')
    # new_model.summary()

    labels = load_label()

    # 모델 사용

    xhat = x_test
    yhat = new_model.predict(xhat)
    #print(yhat[0])
    predictions = np.array([np.argmax(pred) for pred in yhat])
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    s = 0
    #filel = np.array(filel)
    txtpath = output_data_path + "result.txt"
    with open(txtpath, "w") as f:
        for i in predictions:
            f.write(Y[s])
            f.write(" ")
            f.write(rev_labels[i])
            f.write("\n")
            s += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--input_data_path", help=" ")
    parser.add_argument("--output_data_path", help=" ")
    args = parser.parse_args()
    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    main(input_data_path, output_data_path)