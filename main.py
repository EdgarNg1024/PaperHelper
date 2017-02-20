# -*- coding: UTF-8 -*-
# 引入必要的库
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# 构建命令行参数解析并分析参数
# 对应使用方式 python test_grader.py --image images/test_01.png
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# 构建答案字典，键为题目号，值为正确答案
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# 加载图片，将它转换为灰阶，轻度模糊，然后边缘检测。
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# 从边缘图中寻找轮廓，然后初始化答题卡对应的轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None

# 确保至少有一个轮廓被找到
if len(cnts) > 0:
    # 将轮廓按大小降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # 对排序后的轮廓循环处理
    for c in cnts:
        # 获取近似的轮廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 如果我们的近似轮廓有四个顶点，那么就认为找到了答题卡
        if len(approx) == 4:
            docCnt = approx
            break
# 对原始图像和灰度图都进行四点透视变换
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# 对灰度图应用大津二值化算法
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# 在二值图像中查找轮廓，然后初始化题目对应的轮廓列表
cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

# 对每一个轮廓进行循环处理
for c in cnts:
    # 计算轮廓的边界框，然后利用边界框数据计算宽高比
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    print 'w' + str(w), 'h' + str(h)

    # 为了辨别一个轮廓是一个方框，要求它的边界框不能太小，在这里边至少是40个像素，而且它的宽高比要近似于1
    # 为了去掉外面的那个方框,所以轮廓不能太长,小于300像素
    # 轮廓肯定会有4个点
    if ar >= 0.9 and 37 < w < 300:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 4:
            questionCnts.append(c)

color = (0, 0, 255)
cv2.drawContours(paper, questionCnts, -1, color, 3)

# cv2.imshow("Original", image)
# cv2.imshow("exam", thresh)
cv2.imshow("exam", paper)
cv2.waitKey(0)
