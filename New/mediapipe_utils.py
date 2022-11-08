# Import packages
import cv2
import mediapipe as mp


def create_handmodel():
    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7)
    return mp_hands, mp_drawing, hands


def mediapipe_detection(image, model):
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    results = model.process(image)  # Make prediction
    #image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


def draw_landmarks(img, res, mp_hands, mp_drawing):
    mp_drawing.draw_landmarks(
        img,
        res,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
    )