"""

"""
import cv2
import os
import numpy as np
from math import *
"""
读取原始图片
图像尺寸统一裁剪

"""


def img_rotate(img: np.array, degree: int):
    """
    图像旋转 保证 图片不会被裁剪
    :param img:
    :return:
    """
    height, width = img.shape[:2]

    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    return imgRotation


if __name__ == "__main__":
    import os
    path = "D:\picture\\bike"
    path2 = "{0}2".format(path)
    if not os.path.exists(path2):
        os.mkdir(path2)
    for root, dirs, files in os.walk(path):
        for file in files:
            p = "{0}\{1}".format(root, file)
            img = cv2.imread(p)
            if img is None:
                continue
            # 图像旋转 并且 裁剪
            img_0 = img_rotate(img, 0)
            img_0 = cv2.resize(img_0, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("{0}\{1}".format(path2, file), img_0)

            # img_90 = img_rotate(img, 90)
            # img_90 = cv2.resize(img_90, (128, 128), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite("D:\picture\\90\{0}".format(file), img_90)
            # 
            # img_180 = img_rotate(img, 180)
            # img_180 = cv2.resize(img_180, (128, 128), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite("D:\picture\\180\{0}".format(file), img_180)
            # 
            # img_270 = img_rotate(img, 270)
            # img_270 = cv2.resize(img_270, (128, 128), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite("D:\picture\\270\{0}".format(file), img_270)
            # 
            # 
