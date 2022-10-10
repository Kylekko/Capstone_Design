import cv2
import mediapipe as mp
import numpy as np
import time, os
from .mediapipe_utils import *
from ..model.sign_model import SignModel
#https://woochan-autobiography.tistory.com/855
#https://velog.io/@jihyeon9975/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%88%98%EC%96%B4-%EB%8F%99%EC%9E%91-%EC%9D%B8%EC%8B%9D-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0

def landmark_to_array(mp_landmark_list): #third
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results): #second
    """Extract the results of both hands and convert them to a np array of size
    if a hand doesn't appear, return an array of zeros
    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: Two np arrays of size (1, 21 * 3) = (1, nb_keypoints * nb_coordinates) corresponding to both hands
    """
    pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        left_hand = landmark_to_array(results.left_hand_landmarks).reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        right_hand = (
            landmark_to_array(results.right_hand_landmarks).reshape(63).tolist()
        )
    return pose, left_hand, right_hand


def append_landmarks_realtime(result, landmark_list): #first
    dic = {"pose":[],"right_hand":[],"left_hand":[]}
    # Store results
    pose, left_hand, right_hand = extract_landmarks(result)
    dic["pose"], dic["left_hand"], dic["right_hand"] = pose, left_hand, right_hand
    for k, v in dic.items():
        if landmark_list[k] is None:
            landmark_list[k].append(0)
        else:
            landmark_list[k].append(v)
    print(landmark_list)
    return landmark_list

def save_raw_array(landmark_list, actions):
    for action in actions:
        for name, landmarks in landmark_list.items():
            data = np.array(landmarks)
            np.save(os.path.join(f'dataset/{action}', f'raw_{action}_{name}'), data)
            print(name, np.shape(data), "저장완료")

def load_reference_signs(action):
    reference_signs = {"name": [], "sign_model": []}
    path = f'dataset/{action}'

    left_hand_list = np.load(os.path.join(path,f'raw_{action}_left_hand.npy'), allow_pickle=True)
    right_hand_list = np.load(os.path.join(path,f"raw_{action}_right_hand.npy"), allow_pickle=True)

    reference_signs["name"].append('sign_1')
    a= SignModel(left_hand_list, right_hand_list)
    reference_signs["sign_model"].append(a)

    rec_left_hand = a.lh_embedding
    rec_right_hand = a.rh_embedding
    return {"right_hand":rec_right_hand, "left_hand":rec_left_hand}

def clear_list(landmark_list):
    landmark_list["pose"].clear()
    landmark_list["right_hand"].clear()
    landmark_list["left_hand"].clear()
    return landmark_list

def save_seq_array(data, action, hand, seq_length):
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)
    np.save(os.path.join(f'dataset/{action}', f'seq_{action}_{hand}'), full_seq_data)


"""
if __name__=="__main__":
    actions = ['sign2']
    seq_length = 30
    secs_for_action = 5

    # MediaPipe hands model
    mp_hands = mp.solutions.holistic

    mp_drawing = mp.solutions.drawing_utils

    hands = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # For webcam input:
    cap = cv2.VideoCapture(0)

    created_time = int(time.time())
    # os.makedirs('dataset', exist_ok=True)

    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    while cap.isOpened():
        for idx, action in enumerate(actions):
            data = []

            ret, img = cap.read()

            img = cv2.flip(img, 1)

            cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imshow('img', img)
            cv2.waitKey(3000)

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                _, img = cap.read()

                img = cv2.flip(img, 1)
                img, result = mediapipe_detection(img, hands)

                # Create the folder of the sign if it doesn't exists
                os.makedirs("C:/Users/USER/Capstone_Design/data", exist_ok=True)  ########

                # Create the folder of the video data if it doesn't exists
                os.makedirs("C:/Users/USER/Capstone_Design/data/sign1", exist_ok=True)  #########

                append_landmarks_realtime(result)

                draw_styled_landmarks(img, result)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break

            right_data, left_data = load_reference_signs()
            #save_array(data)
            create_data(right_data, action, "right")
            create_data(left_data, action, "left")

            clear_list()
        break
"""
