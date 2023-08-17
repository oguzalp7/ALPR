import cv2
import numpy as np
from alpr_utils import NMS, get_outputs, draw


class VehiclePoseDetector:
    def __init__(self):
        self.model_weights_path = 'FRD_weights/FRNet_YOLOv3_50000.weights'
        self.model_cfg_path = 'FRD_weights/FRNet_YOLOv3.cfg'
        self.model_metadata_path = 'FRD_weights/FRNet.names'

        self.model = cv2.dnn.readNetFromDarknet(self.model_cfg_path, self.model_weights_path)
        with open('FRD_weights/FRNet.names', 'r') as f:
            self.metadata = f.read().strip().split('\n')


    def detect_vehicle_pose(self, image):
        H, W, _ = image.shape
        # convert image
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), True)
        self.model.setInput(blob)
        detections = get_outputs(self.model)
        # bboxes, class_ids, confidences
        bboxes = []
        class_ids = []
        scores = []

        for detection in detections:
            bbox = detection[:4]

            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

            bbox_confidence = detection[4]

            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])

            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

        return NMS(bboxes, class_ids, scores)
    
    def get_best_results(self, image, debug=False):
        bboxes, class_ids, scores = self.detect_vehicle_pose(image)
        best_bbox, best_class_name = None, None
        if len(bboxes) > 0:
            best_bbox = bboxes[0]
            best_class_name = self.metadata[class_ids[0]]
        if debug and best_bbox is not None:
            draw(best_bbox, image)
        elif debug and best_bbox is None:
            pass

        return best_bbox, best_class_name