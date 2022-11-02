import cv2
import mediapipe as mp
import numpy as np
import time, os

from PIL import ImageFont, ImageDraw, Image

actions = ['안녕하세요', '수어', '팀', '입니다', '발표', '시작', '하겠습니다', '감사합니다', '.', '삭제']
seq_length = 10
secs_for_action = 80

font = ImageFont.truetype('./fonts/SCDream6.otf', 20)

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# For webcam input:
cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.text(xy=(10, 20), text=f'Waiting for collecting {action} action...', font=font, fill=(255, 255, 255))

        img = np.array(img)
        cv2.imshow('img', img)
        cv2.waitKey(8000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # RGB -> BGR

            if result.multi_hand_landmarks is not None: # 인식되면 true
                for res in result.multi_hand_landmarks: 
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint # joints로 벡터를 계산해서 각도계산
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint # v2 - v1으로 각도 계산
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 벡터들의 크기를 모두 1로 정규화

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n', # arccos -> cos 역함수
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree # 15개의 각도를 구해서 angle변수에 저장
                                              # angle이 radian으로 나오니까 degree값으로 변환해준다

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(
                        img,
                        res,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                    )

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data)
    break