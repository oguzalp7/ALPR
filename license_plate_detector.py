from ultralytics import YOLO
import cv2
from alpr_utils import find_closest_object

class LicencePlateDetector:
    def __init__(self, min_score=0.2):
        self.model_path = 'license_plate_detector.pt'
        self.model = YOLO(self.model_path)
        self.min_score = min_score

    def detect_licence_plate(self, image, debug=False):
        detections = self.model(image)[0]
        detections = detections.boxes.data.tolist()
        return detections
    
    def find_closest_licence_plate(self, detections):
        return find_closest_object(detections)
    
    def generate_licence_plate_roi(self, image, detections, debug):
        licence_plate = self.find_closest_licence_plate(detections)
        x1, y1, x2, y2, score, class_id = licence_plate
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        if debug:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if score >= self.min_score:
            return image[y1: y2, x1: x2]
        else:
            return None
    