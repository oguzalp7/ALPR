from ultralytics import YOLO
import cv2
from alpr_utils import find_closest_object, filter_detections_by_class, process_plate

class VehicleDetector:
    def __init__(self, min_score=0.2):
        self.model_path = '/home/oguz/Desktop/SASA-CV/yolov8x.pt'
        self.model = YOLO(self.model_path)
        self.vehicle_ids = [2.0, 3.0, 5.0, 7.0]
        self.color = (255, 0, 255)
        self.thickness = 2
        self.min_score = min_score

    def detect_vehicles(self, image):
        detections = self.model(image)[0]
        detections = detections.boxes.data.tolist()
        detections = filter_detections_by_class(detections, self.vehicle_ids)
        return detections
    
    def generate_vehicle_roi(self, image, detections, debug):
        closest_vehicle = find_closest_object(detections)
        x1, y1, x2, y2, score, class_id = closest_vehicle
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        if debug:
            cv2.rectangle(image, (x1, y1), (x2, y2), self.color, self.thickness)
        if score >= self.min_score:
            return image[y1:y2, x1:x2]
        else:
            return None

    def find_closest_vehicle(self, detections):
        return find_closest_object(detections)