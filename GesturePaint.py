import cv2
import mediapipe as mp

from HandsAnalyzer import *
from PaintingTools import *

cap = cv2.VideoCapture(0)

#mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands_analyzer = None

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1000, 800))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame)

        if(hands_analyzer == None):
            hands_analyzer = HandsAnalyzer(mp_hands, results, frame)
        else:
            hands_analyzer.set_attributes(mp_hands, results, frame)

        hands_analyzer.draw_hand()

        if(hands_analyzer.detect_sign() == "INDEX SIGN"):
            index_position = hands_analyzer.calculate_index_position()
            PaintingTools().draw_circle(index_position[0], index_position[1])
            
        PaintingTools().update(frame)

        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Webcam frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()














'''
if results.multi_hand_landmarks:
    for i, hand in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[i]
                
        if handedness.classification[0].label == 'Right':
            index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cv2.circle(frame, (int(index_tip.x*frame.shape[1]), int(index_tip.y*frame.shape[0])), 10, (0, 255, 0), 2)
'''