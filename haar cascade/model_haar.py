# Loads the data required for detecting the license plates from cascade classifier.
import cv2 # pip install opencv-python
from matplotlib import pyplot as plt

plate_cascade = cv2.CascadeClassifier('vn_license_plate.xml')
# add the path to 'vn_license_plate.xml' file.

def detect_plate(img, text=''):  # the function detects and perfors blurring on the number plate.
    # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    plate_rect = plate_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=7)
    plate = img.copy()
    for (x, y, w, h) in plate_rect:
        plate = img[y:y + h, x:x + w, :]
        # finally representing the detected contours by drawing rectangles around the edges.
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 181, 155), 3)

    return plate  # returning the processed image.

img_path = "8.jpg"
img = cv2.imread(img_path)
plate = cv2.imwrite('output_8.jpg', detect_plate(img))   # ten img_path la gi thi output la the
