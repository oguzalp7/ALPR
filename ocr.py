import easyocr
import cv2
from alpr_utils import bbox_to_roi
import numpy as np

class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True, )

    def read_text(self, image):
        ret = ""
        detections = self.reader.readtext(image)
        lst_roi = []
        #c = 0
        for detection in detections:
            text_bbox, text, text_score = detection
            p1, p2, p3, p4 = text_bbox
            x1, y1 = p1
            x2, y2 = p3
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            #cv2.imshow(f'text{c}', bbox_to_roi(image, x1, y1, x2, y2))
            #c += 1
            #print(text, text_score)
            if text_score > 0.2:
                ret += text
        return ret.strip().upper().replace(" ", "")
        