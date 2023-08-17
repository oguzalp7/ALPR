from glob import glob
import cv2
from alpr import ALPR

alpr = ALPR()

def test_on_dataset():
    images = glob('real_data/*.jpeg')
    for image_path in images:
        image = cv2.imread(image_path)
        ret, text = alpr.extract_text(image)
        if ret:
            print(text)

def test_on_realtime():
    camera_src = 0
    cap = cv2.VideoCapture(camera_src)

    while True:
        _, frame = cap.read()
        ret, text = alpr.extract_text(frame)
        if ret:
            print(text)

if __name__ == '__main__':
    test_on_dataset()
    #test_on_realtime()
