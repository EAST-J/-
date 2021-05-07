import cv2
import numpy as np
import imutils
from tracker import *

tracker = EuclideanDistTracker()


if __name__ == "__main__":
    video = cv2.VideoCapture('highway.mp4')
    # b_sub = cv2.bgsegm.createBackgroundSubtractorMOG(history=80)
    b_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    #b_sub = cv2.createBackgroundSubtractorKNN()
    while (True):
        ret, frame = video.read()
        if not ret:
            break
        ###Object detection
        frame = imutils.resize(frame, width=500)

        mask = b_sub.apply(frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5) ###对于mask进行中值滤波，去除离散点的干扰
        img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []  ###define object centers
        for cnt in contours:
            ###计算轮廓点的面积
            area = cv2.contourArea(cnt)
            if area > 80:
                # cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                centers.append([x, y, w, h])
        ###object tracking
        boxes_ids = tracker.update(centers)
        for boxes_id in boxes_ids:
            x, y, w, h, id = boxes_id
            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.waitKey(30)
    video.release()
    cv2.destroyAllWindows()
