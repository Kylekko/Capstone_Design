import numpy as np
import os
import cv2
from mediapipe_utils import *
import numpy as np
import time, os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from threading import Thread
import pyttsx3

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

font = ImageFont.truetype('./fonts/SCDream6.otf', 20)
model = load_model('models/model.h5')
engine = pyttsx3.init()
path_dir = './dataset'
seq_length = 10
secs_for_action = 80

seq = []
action_seq = []
word = []
content = ''

def draw_word(img, x, y, word, color = 0):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), word, font=font, fill=(color, color, color))
    img = np.array(img)
    return img

def compute_angle(res): #create/test둘다
    joint = np.zeros((21, 4))
    for j, lm in enumerate(res.landmark):
        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

    # Compute angles between joints
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
         :3]  # Parent joint # joints로 벡터를 계산해서 각도계산
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
         :3]  # Child joint # v2 - v1으로 각도 계산
    v = v2 - v1  # [20, 3]
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # 벡터들의 크기를 모두 1로 정규화

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',  # arccos -> cos 역함수
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

    angle = np.degrees(angle)  # Convert radian to degree # 15개의 각도를 구해서 angle변수에 저장
    # angle이 radian으로 나오니까 degree값으로 변환해준다
    return angle, joint

def create_seq_data(data, seq_length, action): #create data
    # Create sequence data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)
    np.save(os.path.join('dataset', f'seq_{action}'), full_seq_data)

def get_actions(): #create data/test
    with open("labels.txt", "r", encoding='UTF8') as f:
        stns = f.readlines()
        f1 = lambda x: x.replace('\n', '')
        actions = list(map(f1, stns))
        return actions

actions = get_actions()
action = ""
mp_hands, mp_drawing, hands = create_handmodel()

def create_data(btn_img, cap):
    actions = get_actions()
    mp_hands, mp_drawing, hands = create_handmodel()
    print(actions)
    os.makedirs('dataset', exist_ok=True)
    actions = [actions[0]]
    print(actions)
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()
        img = cv2.flip(img, 1)
        resize_img = cv2.resize(img, (900, 600))
        img = draw_word(resize_img, 10, 20, f'Waiting for collecting {action} action...', 255)
        img = cv2.hconcat([img, btn_img])
        cv2.imshow('video', img)
        cv2.waitKey(5000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            resize_img = cv2.resize(img, (900, 600))
            img, result = mediapipe_detection(resize_img, hands)

            if result.multi_hand_landmarks is not None:  # 인식되면 true
                for res in result.multi_hand_landmarks:
                    angle, joint = compute_angle(res)

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    draw_landmarks(img, res, mp_hands, mp_drawing)
            img = cv2.hconcat([img, btn_img])
            cv2.imshow('video', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}'), data)

        create_seq_data(data, seq_length, action)
        img = draw_word(resize_img, 10, 20, 'create_data완료', 255)
        img = cv2.hconcat([img, btn_img])
        cv2.imshow('video', img)
        cv2.waitKey(2000)

def train_data():
    actions = get_actions()
    data = []
    for name in os.listdir(path_dir):
        if 'seq' in name:
            a = np.load(os.path.join(path_dir, name))
            data.extend(a)

    data = np.array(data)
    x_data = data[:, :, :-1]
    labels = data[:, 0, -1]

    y_data = to_categorical(labels, num_classes=len(actions))

    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.3, random_state=1)  # 7:3

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)

    model = Sequential([
        LSTM(80, activation='relu', input_shape=x_train.shape[1:3]),
        Dropout(0.3),
        Dense(40, activation='relu'),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dense(len(actions), activation='softmax')  # 10
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs= 200,
        callbacks=[
            ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
        ]
    )
    fig, loss_ax = plt.subplots(figsize=(16, 10))
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history.history['acc'], 'b', label='train acc')
    acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.show()
    return history.history['acc'][-1], history.history['val_acc'][-1]


def test_model(btn_img, cap, tts_flag):
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    resize_img = cv2.resize(img, (900, 600))
    img, result = mediapipe_detection(resize_img, hands)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            angle, joint = compute_angle(res)
            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)
            print(np.shape(np.array(seq)))

            if len(seq) > 19: #len(seq) 20으로 유지
                del seq[0]
            draw_landmarks(img, res, mp_hands, mp_drawing)

            if len(seq) < seq_length:
                continue

            # 인퍼런스 한 결과 추출
            input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)

            # 어떠한 인덱스 인지 뽑아낸다
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            print(i_pred, conf)

            # confidence 가 95%이하이면 액션을 취하지 않았다 판단
            if conf < 0.95:
                continue

            action = actions[i_pred]
            action_seq.append(action)
            print(action)

            if len(action_seq) < 20:
                continue

            if action_seq[-1] == action_seq[-3] == action_seq[-5] == action_seq[-7]:
                if action == '삭제':
                    word.clear()
                    tts_flag = False
                else:
                    word.append(action)

        global content
        content = ''
        for i in word:
            if i in content:
                pass  # 중복 출력 방지
            else:
                content += i
                if i == '.': # .는 문장 끝이라고 생각하고 공백제거
                    continue
                else:
                    content += " "

        if not tts_flag: #한번만 음성지원
            if (len(content) > 0) and (content[-1] == '.'):
                Thread(target=tts, args=(content,)).start() #thread로 동시시행
                tts_flag = True

    if len(content)>0: # 단어가 있어야 표시
        img = draw_word(img, 10, 40, content, 255)

    img = draw_word(img, 10, 10, "테스트 중...", 255)
    img = cv2.hconcat([img, btn_img])
    cv2.imshow('video', img)
    return tts_flag

def tts(content):
    engine.setProperty('rate', 120) # 속도조절 default 200
    engine.say(content)
    engine.runAndWait()