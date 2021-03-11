import cv2
from tkinter import messagebox

global pnum
pnum = 0

def model() :
    while(True):

        if cv2.waitKey(1) == ord('c'):
            global pnum
            pnum+=1
            return True
        else :
            return False

def camera() :


    cam = cv2.VideoCapture(0)

    if cam.isOpened() == False:  # 카메라 생성 확인
        print('CAM Error')
        exit()

    print('width :%d, height : %d' % (cam.get(3), cam.get(4)))
    text = 'Smile~'
    # count = 100

    while (True):
        ret, frame = cam.read()  # Read 결과와 frame

        if (ret):

            cv2.imshow('web_cam', frame)

            if model():

                messagebox.showinfo("Taking a picture", "Smile~")
                #cv2.putText(frame, text, (100, 100), cv2.FONT_ITALIC, 1, (255, 0, 0))
                cv2.imwrite('picture' + str(pnum) + '.jpg', frame)

        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


camera()




