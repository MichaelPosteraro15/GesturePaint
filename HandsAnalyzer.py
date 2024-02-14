from itertools import count
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils

class HandsAnalyzer:

    instance = None

    mp_hands = None
    results = None
    frame = None

    def __init__(self, mp_hands, results, frame):
        self.mp_hands = mp_hands
        self.results = results
        self.frame = frame
    
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def set_attributes(self, mp_hands, results, frame):
        self.mp_hands = mp_hands
        self.results = results
        self.frame = frame
        
    # Calculate the contour. If my finger tips are inside this polygon then the hand is closed.
    # The contour contain ...
    def calculate_contour(self, hand):
        points = []

        for i in [0, 1, 3, 6, 10, 14, 18, 17]:
            current_point = hand.landmark[i]
            points.append([int(current_point.x*self.frame.shape[1]), int(current_point.y*self.frame.shape[0])])
                    
        nparray = np.array(points)
        nparray = nparray.reshape((-1, 1, 2))

        return nparray
    
    # Calculate the finger tip position.
    def calculate_tips_position(self, hand):
        points = []

        for i in [4, 8, 12, 16, 20]:
            current_tip = hand.landmark[i]
            points.append([int(current_tip.x*self.frame.shape[1]), int(current_tip.y*self.frame.shape[0])])
            
        return points

    # Check if a particular finger is up. Return true o false.
    def is_open(self, hand, tip):
        contour = self.calculate_contour( hand)
        dist = cv2.pointPolygonTest(contour, (tip[0], tip[1]), False)

        if(dist >= 0):
            return False
        return True

    # Return all finger landmarks (4 lendmarks)
    def calculate_finger(self, hand, j):
        finger_landmarks = []

        for i in range(4):
            finger_landmarks.insert(i, hand.landmark[j-i])

        return finger_landmarks
    
    # Return true if hands are detected.
    def hands_detected(self):
        if self.results.multi_hand_landmarks:
            return True
        
        return False
    
    def calculate_index_position(self):
        tips_position = self.calculate_tips_position(self.calculate_hand())
        return tips_position[1]
        
    def calculate_hand(self):
        if self.hands_detected():
            for i, hand in enumerate(self.results.multi_hand_landmarks):
                handedness = self.results.multi_handedness[i]

                if handedness.classification[0].label == 'Right':
                    return hand
        return None

    # Return all the fingers up.
    def fingers_up(self):
        fingers = [None] * 5

        hand = self.calculate_hand()
        if(hand!= None):
            for j, tip in enumerate(self.calculate_tips_position(hand)):
                if self.is_open(hand, tip):    
                    finger = self.calculate_finger(hand, j)
                    fingers.insert(j, finger)
        
        return fingers
    
    # count how many fingers are open
    def count_open_fingers(self, fingers):
        counter = 0
        for finger in fingers:
            if finger != None:
                counter += 1
        
        return counter
                  
    # Detect victory sign
    def victory_sign(self, fingers):
        if (self.count_open_fingers(fingers)) == 2 and (fingers[1] and fingers[2]):
            return True
        
        return False
    
    # Detect index finger up (useful to draw)
    def index_sign(self, fingers):
        if self.count_open_fingers(fingers) == 1 and fingers[1]:
            return True
        
        return False
    
    # Detect ok sign with thumb up and the other finger closed
    def ok_sign(self, fingers):
        if self.count_open_fingers(fingers) == 1 and fingers[0]:
            return True
        
        return False
    
    # Detect gimme 5 sign aka when user open hand
    def gimme5_sign(self, fingers):
        if self.count_open_fingers(fingers) == 5:
            return True
        
        return False
    
    # Detect fist sign aka when user close hand
    def fist_sign(self, fingers):
        if self.count_open_fingers(fingers) == 0 and self.hands_detected(): 
            return True
        
        return False
    
    # Function to detect which sign user performs
    def detect_sign(self):
        fingers = self.fingers_up()

        if(self.victory_sign(fingers) == True):
            return "V SIGN"
        if(self.index_sign(fingers) == True):
            return "INDEX SIGN"
        if(self.ok_sign(fingers) == True):
            return "OK SIGN"
        if(self.gimme5_sign(fingers) == True):
            return "OPEN HAND SIGN"
        if(self.fist_sign(fingers) == True):
            return "CLOSE HAND SIGN"
        
    # Draw hand landmarks
    def draw_hand_landmarks(self):
        if self.results.multi_hand_landmarks:
            for num, hand in enumerate(self.results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(self.frame, hand, self.mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                        )
        
    # Useful to check if the hand sign if a fist
    # Draw a poligon based on hand landmarks (specifically those that help me to detect if finger are inside the palm)
    def draw_hand_contour(self):
        if self.results.multi_hand_landmarks:
            for i, hand in enumerate(self.results.multi_hand_landmarks):
                handedness = self.results.multi_handedness[i]

                if handedness.classification[0].label == 'Right':
                    contour = self.calculate_contour(hand)

                    cv2.polylines(self.frame, [contour], True, (0, 255, 0), 2)

    # Draw both the hand contour and the landmarks
    def draw_hand(self):
        self.draw_hand_landmarks()
        self.draw_hand_contour()


    
    
                    