import numpy as np
import cv2
import imutils
import easyocr
import string

reader = easyocr.Reader(['en'])

def calculate_bbox_area(detection):
    x1, y1, x2, y2, score, class_id = detection
    width = x2 - x1
    height = y2 - y1
    area = width * height
    return area

def find_closest_object(detections):
    max_area = 0
    closest_obj_bbox = None

    for detection in detections:
        
        area = calculate_bbox_area(detection)
        if area > max_area:
            max_area = area
            closest_obj_bbox = detection

    return closest_obj_bbox

def filter_detections_by_class(detections, interested_class_ids):
    filtered_detections = []

    for detection in detections:
        _, _, _, _, _, class_id = detection
        if class_id in interested_class_ids:
            filtered_detections.append(detection)

    return filtered_detections

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def process_plate(plate_roi):
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize( gray, None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    hist_eq = cv2.equalizeHist(blur)
    cv2.imshow("2",hist_eq)
    
    # gray = cv2.resize( gray, None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # gray = cv2.medianBlur(gray, 3)
    # cv2.imshow('hist', cv2.equalizeHist(blur))
    # ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # thresh = cv2.bitwise_not(thresh)
    # cv2.imshow("x",thresh)
    # detections = reader.readtext(thresh)
    # print(detections)


def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return  cv2.filter2D(image, -1, kernel)

def get_outputs(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)

    outs = [c for out in outs for c in out if c[4] > 0.1]

    return outs

def draw(bbox, image):
    xc, yc, w, h = bbox
    cv2.rectangle(image,
                        (xc - int(w / 2), yc - int(h / 2)),
                        (xc + int(w / 2), yc + int(h / 2)),
                        (160, 255, 0), 2)

    

def NMS(boxes, class_ids, confidences, overlapThresh = 0.5):

    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)

    # Return empty lists, if no boxes given
    if len(boxes) == 0:
        return [], [], []

    x1 = boxes[:, 0] - (boxes[:, 2] / 2)  # x coordinate of the top-left corner
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)  # y coordinate of the top-left corner
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)  # x coordinate of the bottom-right corner
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)  # y coordinate of the bottom-right corner

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        # Create temporary indices
        temp_indices = indices[indices != i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
        xx2 = np.minimum(box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
        yy2 = np.minimum(box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if overlapping greater than our threshold, remove the bounding box
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

    # return only the boxes at the remaining indices
    return boxes[indices], class_ids[indices], confidences[indices]

def bbox_to_roi(image, x1, y1, x2, y2):
    """
    Extracts the Region of Interest (ROI) from the input image defined by the bounding box.

    Parameters:
        image (numpy.ndarray): The input image in NumPy array format (BGR or grayscale).
        x1, y1, x2, y2 (int): Coordinates of the bounding box (top-left and bottom-right).

    Returns:
        numpy.ndarray: The Region of Interest (ROI) extracted from the input image.
    """
    # Ensure coordinates are integers and positive
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])

    # Extract the ROI
    roi = image[y1:y2, x1:x2]

    return roi

def check_turkish_license_plate_format(plate_text):
    """
    Türk plaka standartlarına uyan sivil plakaların karakter sayısı 7 veya 8' dir. 
    Karakter sayısı, plakanın 2. kısmını oluşturan harf sayısıyla doğrudan ilişkilidir. 
    İkinci kısım 1,2 veya 3 harften oluşmaktadır.
    2. kısımda 1 harf varsa 3. kısımda 4 rakam, 2 harf varsa 3 veya 4 rakam, 3 harf varsa 2 rakam olur.
    """
    
    """
    Checks if the extracted text follows the format of Turkish license plates.

    Parameters:
        plate_text (str): The extracted text from the license plate.

    Returns:
        bool: True if the plate text follows the format, False otherwise.
    """
    # Remove any whitespace and convert the text to uppercase for consistent comparison
    plate_text = plate_text.replace(" ", "").upper()

    # Check if the character count is 7 or 8
    if len(plate_text) not in (7, 8):
        return False

    # Get the second part of the license plate (which contains letters)
    second_part = plate_text[3:6]

    # Get the third part of the license plate (which contains numbers)
    third_part = plate_text[6:]

    # Check the relationship between the number of letters in the second part and the number of digits in the third part
    if len(second_part) == 1:
        if len(third_part) != 4 or not third_part.isdigit():
            return False
    elif len(second_part) == 2:
        if len(third_part) not in (3, 4) or not third_part.isdigit():
            return False
    elif len(second_part) == 3:
        if len(third_part) != 2 or not third_part.isdigit():
            return False
    else:
        return False

    # All checks passed, the plate text follows the format
    return True

