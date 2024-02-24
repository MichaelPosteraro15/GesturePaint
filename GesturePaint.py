import cv2
import mediapipe as mp

from HandsAnalyzer import *

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands_analyzer = None

imgCanvas = np.zeros((600, 800, 3), np.uint8)

xp,yp = 0, 0

with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85) as hands:
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (800, 600))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame)

        if(hands_analyzer == None):
            hands_analyzer = HandsAnalyzer(mp_hands, results, frame)
        else:
            hands_analyzer.set_attributes(mp_hands, results, frame)

        hands_analyzer.draw_hand()

        if(hands_analyzer.detect_sign() == "INDEX SIGN"):
            index_position = hands_analyzer.calculate_index_position()
            x,y = index_position[0], index_position[1]

            if(xp == 0 and yp == 0):
                xp,yp = x, y
        
            cv2.line(imgCanvas, (xp, yp), (x,y), (255,  255,  255), 15)

            xp, yp = x, y
        elif(hands_analyzer.detect_sign() == "V SIGN"):
            index_position = hands_analyzer.calculate_index_position()
            x,y = index_position[0], index_position[1]

            if(xp == 0 and yp == 0):
                xp,yp = x, y
        
            cv2.line(imgCanvas, (xp, yp), (x,y), (0,  0,  0), 30)

            xp, yp = x, y
        else:
            xp, yp = 0, 0

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, imgCanvas)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Webcam frame", frame)
     
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
