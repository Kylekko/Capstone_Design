#import sys
#print(sys.path)
#sys.path.append("C:/Users/USER/Desktop/A/New/model")
#print(sys.path)
from utils.mediapipe_utils import *
from model.sign_model import SignModel
from utils.landmark_utils import *

actions = ['sign1', 'sign2']
seq_length = 30
secs_for_action = 5

if __name__=="__main__":
    # MediaPipe hands model
    mp_hands = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # For webcam input:
    cap = cv2.VideoCapture(0)

    created_time = int(time.time())

    # Create the folder of the sign if it doesn't exists
    for action in actions:
        os.makedirs(f"dataset/{action}", exist_ok=True) 

    landmark_list = {"right_hand": [], "left_hand": [], "pose": []}
    while cap.isOpened():
        for idx, action in enumerate(actions):
            data = []

            ret, img = cap.read()
            cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imshow('img', img)
            cv2.waitKey(3000)

            start_time = time.time()
            while time.time() - start_time < secs_for_action:
                _, img = cap.read()
                img, result = mediapipe_detection(img, hands) #extract landmarks

                landmark_list = append_landmarks_realtime(result, landmark_list) #save the landmarks in the dic(list)

                draw_styled_landmarks(img, result) #draw landmarks

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break

            save_raw_array(landmark_list, actions) #save landmarks
            data = load_reference_signs(action) #create embedded data with raw data

            # Create sequence data
            for action in actions:
                #landmark_list key순서가 변경될때마다 담기는 값이 달라짐.e)pose인데 right_hand담김.
                for (name, landmarks), key in zip(landmark_list.items(), data.keys()):
                    print("+++++++++++++++++++++++++++",name)
                    print("\n\n",data[key],key, action,name,"\n\n~~~~~~~~~~~~~~~~\n")
                    save_seq_array(data[key], action, name, seq_length) #create sequence data with embedded data

            landmark_list = clear_list(landmark_list) #clear dic(list)
        break
