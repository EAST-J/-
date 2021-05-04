import numpy as np
import cv2
import imutils
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, default="images/scottsdale")
ap.add_argument("-o", "--output", type=str, required=True)
ap.add_argument("-c", "--crop", type=int, default=0)
args = vars(ap.parse_args())

imagepaths = sorted(list(paths.list_images(args["images"])))
images = []

for path in imagepaths:
    img = cv2.imread(path)
    images.append(img)

stitcher = cv2.createStitcher()
(status, stitched) = stitcher.stitch(images)
if status == 0:
    if args["crop"] > 0:
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype='uint8')
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        minRect = mask.copy()
        sub = mask.copy()
        while cv2.countNonZero(sub) > 0:
            minRect = cv2.erode(minRect, None)
            sub = cv2.subtract(minRect, thresh)
        # find contours in the minimum rectangular mask and then
        # extract the bounding box (x, y)-coordinates
        cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)

        # use the bounding box coordinates to extract the our final
        # stitched image
        stitched = stitched[y:y + h, x:x + w]
        cv2.imwrite(args["output"], stitched)
