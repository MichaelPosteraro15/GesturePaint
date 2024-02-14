import cv2
import mediapipe as mp



#mp_drawing = mp.solutions.drawing_utils

class PaintingTools:
    instance = None

    points = []

    def __init__(self):
        pass

    def draw_circle(self, x, y):
        self.points.append((x,y))
        #cv2.circle(frame, (x, y),  50, (255,  255,  255), -1)

    def update(self, frame):
        for point in self.points:
            #cv2.circle(frame, (int(point[0]), int(point[1])),  10, (255,  255,  255), -1)
            cv2.line(frame, (int(point[0]), int(point[1])), (int(point[0]+1), int(point[1]+1)), (255,  255,  255),  10)


    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance