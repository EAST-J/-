import cv2
import numpy as np
import imutils

lower_red = np.array((0, 43, 46))
upper_red = np.array((10, 255, 255))

if __name__ == "__main__":
    video = cv2.VideoCapture('highway.mp4')
    #b_sub = cv2.bgsegm.createBackgroundSubtractorMOG(history=80)
    b_sub = cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=50,detectShadows=False)
    while (True):
        ret, frame = video.read()
        if not ret:
            break
        ###Object detection
        frame = imutils.resize(frame, width=500)

        mask = b_sub.apply(frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        # mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         mask[i, j] = mask[i, j] and mask_red[i, j]
        img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            ###计算轮廓点的面积
            area = cv2.contourArea(cnt)
            if area > 100:
                # cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        # cv2.imshow('maskred', mask_red)
        cv2.waitKey(30)
    video.release()
    cv2.destroyAllWindows()
