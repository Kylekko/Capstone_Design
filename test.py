import cv2
import mediapipe as mp
import numpy as np

from time import sleep

from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import load_model

actions = ['안녕하세요', '수어', '팀', '입니다', '발표', '시작', '하겠습니다', '감사합니다', '.', '삭제']
seq_length = 10

model = load_model('models/model.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = ImageFont.truetype('./fonts/SCDream6.otf', 20)

seq = []
action_seq = []
word = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # v 정규화

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian -> degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(
                img,
                res,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
            )

            if len(seq) < seq_length:
                continue

            # 인퍼런스 한 결과 추출
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            # 어떠한 인덱스 인지 뽑아낸다
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            # confidence 가 95%이하이면 액션을 취하지 않았다 판단
            if conf < 0.95:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 10:
                continue

            if action_seq[-1] == action_seq[-3] == action_seq[-5] == action_seq[-7] == action_seq[-9]:
                if action == '삭제':
                    word.clear()
                # elif action == '.':
                #     draw.text(xy=(10, 20), text="Save Message", font=font, fill=(255, 255, 255))
                #     sleep(2)
                #     word.append(UI에 담을 리스트)
                #     word.clear()
                else:
                    word.append(action)

            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)

            img = np.array(img)

        content = ''
        for i in word:
            if i in content:
                pass    # 중복 출력 방지
            else:
                content += i
                content += " "

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.text(xy=(10, 20), text=content, font=font, fill=(255, 255, 255))
        img = np.array(img)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break