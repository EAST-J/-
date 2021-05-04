import numpy as np
import imutils
import cv2
import math

img_path = 'D:/360MoveData/Users/JSJ/Desktop/scottsdale/'
class Stitcher:
    def __init__(self):
        # 检测OpenCV的版本
        self.isv3 = imutils.is_cv3()
    def stitch(self, images, ratio=0.4, reprojThresh=4.0,showMatches=False,Edgeprocessing=False):
        #实现两张图像的拼接
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # 检测两张图片里的关键点、提取局部不变特征。
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        #匹配两张图片里的特征
        if M is None:
            return None
        #如果返回匹配的M为None，就是因为现有的关键点不足以匹配生成全景图。
        #假设M不返回None，拆包返回元组，包含关键点匹配matches、从RANSAC算法中得到的最优单映射变换矩阵H以及最后的单映计算状态列表status，用来表明那些已经成功匹配的关键点。
        (matches, H, status) = M
        if not Edgeprocessing:
            result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        else:
            result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            rows, cols = imageB.shape[:2]
            for col in range(0, cols):
                if imageB[:, col].any() and result[:, col].any():  # 开始重叠的最左端
                    left = col
                    break
            for col in range(cols - 1, 0, -1):
                if imageB[:, col].any() and result[:, col].any():  # 重叠的最右一列
                    right = col
                    break
            res = np.zeros([rows, cols, 3], np.uint8)
            for row in range(0, rows):
                for col in range(0, cols):
                    if not imageB[row, col].any():  # 如果没有原图，用旋转的填充
                        res[row, col] = result[row, col]
                    elif not result[row, col].any():
                        res[row, col] = imageB[row, col]
                    else:
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(imageB[row, col] * (1 - alpha) + result[row, col] * alpha, 0, 255)
            result[0:imageB.shape[0], 0:imageB.shape[1]] = res
        #根据是否边缘处理的指令返回缝合后的图像
        # 决定是否可视化匹配结果
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)
            return (result, vis)
        #将两张图片关键点的匹配可视化
        return result
    # 接收照片，检测关键点和提取局部不变特征
    # 用到了高斯差分（Difference of Gaussian (DoG)）关键点检测，和SIFT特征提取
    # detectAndCompute方法用来处理提取关键点和特征
    # 返回一系列的关键点
    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# 将图像转换为灰度图
        #根据不同版本进行不同的操作
        #检测是否用了OpenCV3.X，如果是，就用cv2.xfeatures2d.SIFT_create方法来实现DoG关键点检测和SIFT特征提取。detectAndCompute方法用来处理提取关键点和特征。
        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        #OpenCV2.4，则cv2.FeatureDetector_create方法来实现关键点的检测（DoG）。detect方法返回一系列的关键点。
        # 用SIFT关键字来初始化cv2.DescriptorExtractor_create，设置SIFT特征提取。调用extractor的compute方法返回一组关键点周围量化检测的特征向量。
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
        #关键点从KeyPoint对象转换为NumPy数组后返回给调用函数。
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    # matchKeypoints方法需要四个参数，第一张图片的关键点和特征向量，第二张图片的关键点特征向量，David Lowe’s ratio测试变量和RANSAC重投影门限。
    #返回值为matches, H, status。分别为匹配的关键点matches，最优单映射变换矩阵 H（3x3)，单映计算的状态列表status用于表示已经成功匹配的关键点。
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # OpenCV已经内置了cv2.DescriptorMatcher_create方法，用来匹配特征。BruteForce参数表示我们能够更详尽计算两张图片直接的欧式距离，以此来寻找每对描述子的最短距离。
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        #knnMatch方法是K = 2的两个特征向量的k - NN匹配（k - nearest neighbors algorithm，K近邻算法），表明每个匹配的前两名作为特征向量返回。
        # 之所以我们要的是匹配的前两个而不是只有第一个，是因为我们需要用David Lowe’s ratio来测试假匹配然后做修剪。
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # 在原始的匹配上循环
        for m in rawMatches:
            #运用 Lowe’s ratio测试特别的来循环rawMatches，用来确定高质量的特征匹配。
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # 计算两串关键点的单映性需要至少四个匹配。为了获得更为可信的单映性，我们至少需要超过四个匹配点。
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # 计算两组点之间的单映
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            return (matches, H, status)
        return None
    # 连线画出两幅图的匹配
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis
#柱面投影函数
def cylindrical_projection(i,f):
    #filename = [str(1 + k) + '.jpg' for k in range(3)]
    filename = img_path + str(i+1)+'.jpg'
    img = cv2.imread(filename)
    rows = img.shape[0]
    cols = img.shape[1]
    # f = cols / (2 * math.tan(np.pi / 8))
    blank = np.zeros_like(img)
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    for y in range(rows):
        for x in range(cols):
            theta = math.atan((x - center_x) / f)
            point_x = int(f * math.tan((x - center_x) / f) + center_x)
            point_y = int((y - center_y) / math.cos(theta) + center_y)
            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                blank[y, x, :] = img[point_y, point_x, :]
    cv2.imwrite(filename,blank)
#与类中的匹配函数类似，用0和1反馈匹配结果，方便打印分组
def classify(imageA,imageB,ratio=0.75):
    sift = cv2.xfeatures2d.SIFT_create()
    (kpsA, featuresA) = sift.detectAndCompute(imageA, None)
    (kpsB, featuresB) = sift.detectAndCompute(imageB, None)
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches=[]
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    print(len(matches))
    if len(matches) > 100:

        return 1
    else:
        return 0
#分组打印函数
#首先将九个jpg文件导入，两两匹配，将匹配结果变换成一个9*9的judge矩阵
#对judge矩阵进行处理，当一个jpg的匹配结果中除自己以外出现两个不同匹配时，我们可以断定这个jpg文件为中间图（三张合并全景图过程中中间的那一张）
#根据三组中间图的匹配结果打印分组信息
#缺陷在于头尾两张图片的顺序不能互换，否则得不到应有的匹配结果
def Matchingprint():
    imagename = [str(1 + i) + '.jpg' for i in range(9)]
    judge = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):
            judge[i][j] = classify(cv2.imread(imagename[i]), cv2.imread(imagename[j]))
    #print(judge)
    Matchingresults = []
    for i in range(9):
        num = 0
        line = []
        for j in range(9):
            if judge[i][j] == 1:
                num += 1
                line.append(j + 1)
        if num == 3:
            if line[1] != i + 1 & line[0] == i + 1:
                line[0] = line[1]
                line[1] = i + 1
            if line[1] != i + 1 & line[2] == i + 1:
                line[2] = line[1]
                line[1] = i + 1
            Matchingresults.append(line)
    return Matchingresults
#考虑到图像拼接的原理，合并的顺序应该是中间图片先与右边图片合并，接着再与左边图片合并
def Drawresult(Matchingresults,i,Is_cylindrical):
    imagename = [str(1 + k) + '.jpg' for k in range(9)]
    m = Matchingresults[i][1]-1
    n = Matchingresults[i][2]-1
    p = Matchingresults[i][0]-1
    imageA = cv2.imread(imagename[m])
    imageB = cv2.imread(imagename[n])
    imageC = cv2.imread(imagename[p])
    stitcher = Stitcher()
    middle = stitcher.stitch([imageA, imageB], Edgeprocessing=True)
    result = stitcher.stitch([imageC, middle], Edgeprocessing=True)
    if not Is_cylindrical:
        cv2.imwrite('original_result'+str(i+1)+ '.jpg',result)
    else:
        cv2.imwrite('cylindrical_result' + str(i + 1) + '.jpg', result)
if __name__ == '__main__':
    #Is_cylindrical = False
    # Is_cylindrical = True
    # if Is_cylindrical ==True:
    #     for i in range(3):
    #         cylindrical_projection(i, 600)
    #     Matchingresults = Matchingprint()
    #     print(Matchingresults)
    # else:
    #     Matchingresults = Matchingprint()
    #     print(Matchingresults)
    # Matchingresults = np.array(Matchingresults)
    # Drawresult(Matchingresults,0,Is_cylindrical)
    # Drawresult(Matchingresults,1,Is_cylindrical)
    # Drawresult(Matchingresults,2,Is_cylindrical)
#———————————————————————————————————————————————————————————————
#将两张图片拼接并展示结果图（进行边缘处理之后的结果图）
#为了更快的处理速度，可以将两张图片统一宽度

    imageA = cv2.imread(img_path+'1.jpg')
    imageB = cv2.imread(img_path+'2.jpg')
    classify(imageA,imageB,ratio=0.4)
#imageA = imutils.resize(imageA, width=400)
#imageB = imutils.resize(imageB, width=400)
    showMatches=True #展示两幅图像特征的匹配,返回vis
    stitcher = Stitcher()
    (direct, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    result = stitcher.stitch([imageA, imageB], Edgeprocessing=True)
    print('first_Finished')
    #imageA = cv2.imread(img_path + '3.jpg')
    #(direct, vis) = stitcher.stitch([imageA, result], showMatches=True)
    # result = stitcher.stitch([imageA, result], Edgeprocessing=True)
#连线图与结果图保存到工程文件夹，这里只显示结果图
    cv2.imwrite('test_vis.jpg', vis)
    cv2.imwrite('test_direct.jpg', direct)
    cv2.imwrite('test_result.jpg', result)
    print('Finished')
    #cv2.imshow('test_result',result)
    #cv2.waitKey(5000)
# ———————————————————————————————————————————————————————————————
