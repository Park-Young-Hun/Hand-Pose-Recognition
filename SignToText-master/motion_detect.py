from keras.models import Model, load_model
import cv2
import numpy as np
from hand_tracker import handtracking

model = load_model('HPRmodelv2.h5') # 모델 연결.

def motion_detect():
    recur = True

    while recur:
        recur = False
        handtracking()


        #일단 비디오 캡쳐 열어서 mp4형태로 저장.
        cap = cv2.VideoCapture('outputWebCan.mp4') #빨간손

        fnum = 0 # frame number지정.
        testdata = [] #testdata를 이용해 비교
        predictions = [] # 예측값을 리스트로 사용할 것.
        imgdata = [] #testdata에 일정 길이마다 append할 것임.

        #웹캡이 열였을 때 조건사항
        while (cap.isOpened() and fnum<=100):
            ret, frame = cap.read()

            image = frame
            try:
                result = image.copy()
            except:
                print("이미지 커비 에러")
                break

            imgdata.append(result)

            if len(imgdata) == 16:
                testdata.append(imgdata)
                imgdata = []

            fnum +=1

        testdata = np.array(testdata)










if __name__ == "__main__":
    motion_detect()