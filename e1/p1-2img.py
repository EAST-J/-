import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

img_path = 'D:/360MoveData/Users/JSJ/Desktop/scottsdale/'


def detectAndDescribe(img, method='sift'):
    """
    输入参数：图像和选择的特征点提取的算法
    返回：关键点和描述符
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'ORB':
        descriptor = cv2.ORB_create()
    else:
        print('method error')
        return
    (kps, features) = descriptor.detectAndCompute(gray, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)


###测试关键点检测部分的代码
# img = cv2.imread(img_path + '1.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# start_time = time.time()
# (kps, features) = detectAndDescribe(gray, method='ORB')
# img = cv2.drawKeypoints(gray, kps, img)
# end_time = time.time()
# print("运行时间：" + str(end_time - start_time))
# cv2.imwrite('result.jpg', img)

def createMatcher(method='sift', crossCheck=True):  ###使用暴力匹配法
    """

    :param method:
    :param crossCheck:True对于匹配的条件会更加严格
    :return:创建匹配关系
    """
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def createMatcher_knn(kps1, kps2, features1, features2, ratio=0.4, reprojThresh=5.0, count_point=4):
    '''

    :param kps1:
    :param kps2:
    :param features1:
    :param features2:
    :param ratio:
    :param reprojThresh:重投影的错误阈值 一般范围为1-10
    :return:
    '''
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    rawMatches = matcher.knnMatch(features1, features2, 2)
    matches = []
    for n in rawMatches:
        if len(n) == 2 and n[0].distance < n[1].distance * ratio:
            matches.append((n[0].trainIdx, n[0].queryIdx))  ###认为是匹配的点

    if len(matches) > count_point:
        ptsA = np.float32([kps1[i] for (_, i) in matches])
        ptsB = np.float32([kps2[i] for (i, _) in matches])
        # 计算两组点之间的单映
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    return None


def classify(img1, img2, ratio=0.4, num=100):
    (kps1, features1) = detectAndDescribe(img1)
    (kps2, features2) = detectAndDescribe(img2)
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    rawMatches = matcher.knnMatch(features1, features2, 2)
    matches = []
    for n in rawMatches:
        if len(n) == 2 and n[0].distance < n[1].distance * ratio:
            matches.append((n[0].trainIdx, n[0].queryIdx))  ###认为是匹配的点
    if len(matches) > num:
        print(len(matches))
        return 1
    else:
        return 0


from PIL import Image
from numpy import average, dot, linalg


# 对图片进行统一化处理
def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image


# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res


# img1 = cv2.imread(img_path + '1.jpg')
# img2 = cv2.imread(img_path + '2.jpg')
img1 = Image.open(img_path + 'test_images/6.jpg')
img2 = Image.open(img_path + 'test_images/5.jpg')
cos = image_similarity_vectors_via_numpy(img1, img2)
print('两图像的余弦相似度为:'+str(cos))
# print(classify(img1, img2))
# (kps1, features1) = detectAndDescribe(img1)
# (kps2, features2) = detectAndDescribe(img2)  ####计算img中的特征点和描述子
# M = createMatcher_knn(kps1, kps2, features1, features2)
# (matches, H, status) = M
# result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
# result[0:img2.shape[0], 0:img2.shape[1]] = img2
# cv2.imshow('test', result)
# cv2.imwrite('result.jpg', result)
# cv2.waitKey(0)
