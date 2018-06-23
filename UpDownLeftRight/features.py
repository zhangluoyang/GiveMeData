"""
使用 python opencv 提取图像特征
这里面有一个自己内部实现的hog特征提取器
代码来源与
重在理解，实际的项目当中直接使用opencv自带的hog特征提取
undo 支持增加噪声生成样本的功能
"""
import cv2
import os
import numpy as np
import math
import matplotlib.pylab as plt


class HogDescriptor(object):

    def __init__(self, img, cell_size: int=16, bin_size: int=9):
        img = np.sqrt(img/np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360/self.bin_size

    def extract(self):
        """
        提取 图像的hog特征
        :return:
        """
        height, width = self.img.shape
        # 计算全局梯度 以及梯度方向
        gradient_magnitude, gradient_angle = self.global_gradient()
        # 绝对值
        gradient_magnitude = abs(gradient_magnitude)
        # 定义一个array用于存放hog特征
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                # 遍历每一个cell
                # 统计cell当中的梯度以及方向
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size, j * self.cell_size:(j + 1) * self.cell_size]
                # 计算得到的一个cell的梯度方向直方图
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        hog_vector = []
        # 统计block的梯度信息 这里的block信息是2*2形式 我们可以对齐修改 支持任意的格式
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                block_vector = []
                # 4个cell特征的融合
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                # 我们对block的梯度统计归一化
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                # 计算分母部分 (均方和)
                magnitude = mag(block_vector)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
        return hog_vector, hog_image

    def global_gradient(self):
        """
        梯度计算 全局
        :return:
        """
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        """
        根据一个cell当中的梯度信息以及角度方向信息统一出这个cell当中的直方图
        :param cell_magnitude:
        :param cell_angle:
        :return:
        """
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                # 遍历每一个cell当中的方块数据
                # 梯度强度
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                # 余数越多表示对max_angle的贡献越大 相反对min_angle的共现越大
                # 这里的统计方式不同 强度*贡献度
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        """
        返回最近的桶
        :param gradient_angle:
        :return:
        """
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        """
        画出每一个cell部分的方向直方图
        :param image:
        :param cell_gradient:
        :return:
        """
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


def getPaths(path: str):
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            paths.append("{0}/{1}".format(root, file))
    return paths


# 我们定义一个函数 用于取到图片数据 并且找到label
# 然后我们把对应的数据 放在txt文件里面
def getData():
    file = open("datas.txt", "w", encoding="utf-8")
    lablePath0 = "D:/picture/bike2"
    labelPath1 = "D:/picture/ship2"
    labelPath2 = "D:/picture/cart2"
    # labelPath3 = "D:/picture/270"

    data_0 = getPaths(lablePath0)
    for data in data_0:
        file.write("{0}\t{1}\n".format(0, data))

    data_1 = getPaths(labelPath1)
    for data in data_1:
        file.write("{0}\t{1}\n".format(1, data))

    data_2 = getPaths(labelPath2)
    for data in data_2:
        file.write("{0}\t{1}\n".format(2, data))
    #
    # data_3 = getPaths(labelPath3)
    # for data in data_3:
    #     file.write("{0}\t{1}\n".format(3, data))

    file.close()


def test():
    img = cv2.imread("./90000.jpg", cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    hog = HogDescriptor(img=img, cell_size=4, bin_size=12)
    vector, image = hog.extract()
    print(np.array(vector).shape)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


# 定义一个函数 用于提取图片的特征
def getFeatures(path: str):
    winSize = (40, 40)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (5, 5)
    nbins = 9
    winStride = (10, 10)
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    files = open(path, "r", encoding="utf-8")
    labels = []
    features = []
    for file in files:
        label, p = file.split("\t")
        img = cv2.imread(p.strip(), cv2.IMREAD_GRAYSCALE)
        hist = hog.compute(img, winStride, padding=(0, 0))
        labels.append(int(label))
        img_shape = np.shape(hist)
        a = np.resize(hist, new_shape=(img_shape[0]))
        del img
        features.append(a.tolist())
    return labels, features


# blockSize.width % cellSize.width == 0 && blockSize.height % cellSize.height == 0
if __name__ == "__main__":
    # getData()
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.cross_validation import train_test_split
    labels, features = getFeatures("./datas.txt")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier

    # model = AdaBoostClassifier(SVC(kernel="linear", C=2.5), n_estimators=10, algorithm="SAMME")
    # model.fit(X=X_train, y=y_train)
    # print(model.score(X=X_test, y=y_test))

    # model = SVC(kernel="linear", C=2.5)
    # model.fit(X=X_train, y=y_train)
    # print(model.score(X=X_test, y=y_test))
    #
    model = GradientBoostingClassifier(max_depth=10, min_samples_leaf=3, n_estimators=128)
    model.fit(X=X_train, y=y_train)
    print(model.score(X=X_test, y=y_test))
