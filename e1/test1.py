import numpy as np
import cv2
from imutils import paths
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, help="Images Input Folder", default="./images/scottsdale")
ap.add_argument("-o", "--output", type=str, required=True, help="Output Folder", default="")

args = vars(ap.parse_args())

imagepaths = sorted(list(paths.list_images(args["images"])))
images = []

for path in imagepaths:
    img = cv2.imread(path)
    images.append(img)

sticher = cv2.createStitcher()
(status,stitched) = sticher.stitch(images)
if status==0:
    cv2.imwrite(args["output"],stitched)
    cv2.imshow("result",stitched)
    cv2.waitKey(0)
else:
    print("FAIL")

