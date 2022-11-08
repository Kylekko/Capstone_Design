import cv2
from screen_func import *
from mediapipe_utils import *
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
#데이터수집, 학습, 테스트

flag = 0

def btn_screen():
    btn_img = np.ones((600,150,3), np.uint8)*255
    btn_img = draw_word(btn_img, 27, 170, "데이터수집")
    btn_img = draw_word(btn_img, 30, 270, "학습/모델")
    btn_img = draw_word(btn_img, 45, 370, "테스트")
    btn_img = draw_word(btn_img, 55, 470, "중지")

    cv2.rectangle(btn_img, (20, 150, 110, 60), (0, 0, 0), 1)
    cv2.rectangle(btn_img, (20, 250, 110, 60), (0, 0, 0), 1)
    cv2.rectangle(btn_img, (20, 350, 110, 60), (0, 0, 0), 1)
    cv2.rectangle(btn_img, (20, 450, 110, 60), (0, 0, 0), 1)
    return btn_img

def click_print(name,flag, param):
    global cap
    print(name,"출력됨",flag)
    if flag == 1:
        create_data(param, cap)

    elif flag == 2:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        resize_img = cv2.resize(img, (900, 600))
        img = draw_word(resize_img, 10, 20, '학습 중..', 255)
        full_img = cv2.hconcat([img, param])
        cv2.imshow('video', full_img)
        cv2.waitKey(1000)
        acc, val_acc = train_data()
        img = draw_word(resize_img, 10, 20, f'acc: {acc}', 255)
        img = draw_word(img, 10, 50, f'val_acc: {val_acc}', 255)
        full_img = cv2.hconcat([img, param])
        cv2.imshow('video', full_img)
        cv2.waitKey(5000)

    elif flag ==3:
        while True:
            test_model(param, cap)
            if (cv2.waitKey(1) == ord('q')) or (flag == 0):
               break

    elif flag == 0:
        screen()

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        print(x,y)
        if (920<x and x<1030) and (150<y and y<210):
            flag = 1
            click_print("데이터수집",flag, param)
        if (920<x and x<1030) and (250<y and y<310):
            flag = 2
            click_print("학습/모델",flag,param)
        if (920<x and x<1030) and (350<y and y<410):
            flag = 3
            click_print("테스트",flag,param)
        if (920<x and x<1030) and (450<y and y<510):
            flag = 0
            click_print("중지", flag, param)

def screen():
    btn_img = btn_screen()

    mp_hands, mp_drawing, hands = create_handmodel()

    # For webcam input:
    global cap
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        resize_img = cv2.resize(img, (900, 600))
        img, result = mediapipe_detection(resize_img, hands)

        if result.multi_hand_landmarks is not None:  # 인식되면 true
            for res in result.multi_hand_landmarks:
                draw_landmarks(img, res, mp_hands, mp_drawing)

        full_img = cv2.hconcat([img,btn_img])

        cv2.namedWindow("video")
        cv2.setMouseCallback("video", mouse_event, btn_img)#btn_img
        cv2.imshow('video', full_img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()

if __name__=="__main__":
    screen()