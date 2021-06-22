# -*- coding:UTF-8 -*-

"""
文件说明：
    关键点检测
"""

try:
    import cv2 as cv
    import numpy as np
    import matplotlib.pyplot as plt
except ModuleNotFoundError as r:
    print('错误原因是： ' + str(r))

class FindPoints(object):
    """
    可通过 FindPoints.__doc__查看
    文件使用说明：
        Input:图像的路径
        return:1、关键的坐标，以列表形式返回[ [[x1,x2,x3...],[y1,y2,y3...]],
                                         [[x1,x2,x3...],[y1,y2,y3...]],
                                        ]
        return:2、输入图像的高

        函数 IsTrangleOrArea 计算三角形的面积
        函数 IsInside 判断某点是否在三角形区域内
        函数 area_optimize 面积阈值优化

    使用方法如下：
        findPoint = FindPoints(img_path)
        函数 convexity_func：功能是凸点检测
            points, h = findPoint.convexity_func()
        函数 pit_func ： 功能是凸点结合凹点进行检测
            points,h = findPoint.pit_func()
        函数 optimize_pit_func 优化后的凹凸点检测
            points,h = findPoint.optimize_pit_func()
        函数 plt_point 绘图
            plt_point(points, name='op_pit',scatter=False)
                points:坐标列表
                name：绘图名称
                scatter：默认为False，指是否绘制散点图
    """
    def __init__(self, img_path):
        self.img_path = img_path
        self.points = []

    def __del__(self):
        self.points = []

    def convexity_func(self):
        """
        凸性点检查
        """
        self.points = []
        img = cv.imread(self.img_path)
        try:
            if img.all is None:
                print('读取的文件不存在！')
        except AttributeError as r:
            print('读取的文件不存在！' + str(r))

        if len(img.shape) == 3:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_gray = img
        res = cv.findContours(img_gray, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        if len(res) == 2:
            contours, _ = res
        else:
            _, contours, _ = res
        for i in range(len(contours)):
            x = []
            y = []
            hull = cv.convexHull(contours[i])
            for i in range(len(hull)):
                x.append(hull[i][0][0])
                y.append(hull[i][0][1])
            # x = np.array(x)
            #             # y = np.array(y)
            self.points.append([x, y])
        return self.points, img_gray.shape[1]

    def pit_func(self):
        """
        凹点检测
        """
        self.points = []
        img = cv.imread(self.img_path)
        if img.all is None:
            raise FileNotFoundError('读取失败！')
        if len(img.shape) == 3:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_gray = img
        res = cv.findContours(img_gray, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        if len(res) == 2:
            contours, _ = res
        else:
            _, contours, _ = res
        for i in range(len(contours)):
            x = []
            y = []
            tu = []
            # ao = []
            hull = cv.convexHull(contours[i], returnPoints=False)
            defects = cv.convexityDefects(contours[i], hull)
            for n in range(len(hull)):
                tu.append(hull[n][0])
            try:
                for m in range(len(defects)):
                    s, e, f, d = defects[m, 0]
                    tu.append(f)
                    # ao.append(f)
            except Exception:
                pass
            tu.sort()
            # ao.sort()
            for j in tu:
                x.append(contours[i][j][0][0])
                y.append(contours[i][j][0][1])
            # x.append(x[0])
            # y.append(y[0])
            self.points.append([x, y])
        return self.points, img_gray.shape[1]

    def IsTrangleOrArea(self, x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    def IsInside(self, x1, y1, x2, y2, x3, y3, x, y):

        # 三角形ABC的面积
        ABC = self.IsTrangleOrArea(x1, y1, x2, y2, x3, y3)

        # 三角形PBC的面积
        PBC = self.IsTrangleOrArea(x, y, x2, y2, x3, y3)

        # 三角形ABC的面积
        PAC = self.IsTrangleOrArea(x1, y1, x, y, x3, y3)

        # 三角形ABC的面积
        PAB = self.IsTrangleOrArea(x1, y1, x2, y2, x, y)
        if (PBC == 0) or (PAC == 0) or (PAB == 0):
            return False

        return (ABC == PBC + PAC + PAB)

    def area_optimize(self, cnt, ao, tu):
        ou = []
        for i in range(len(ao)):
            count = 0
            for j in range(len(tu)):
                if ao[i] < tu[j]:
                    x1, y1 = cnt[tu[j - 1]][0]
                    x2, y2 = cnt[tu[j]][0]
                    x3, y3 = cnt[ao[i]][0]
                    for m in range(tu[j - 1], tu[j] - 1):
                        x, y = cnt[m][0]
                        if self.IsInside(x1, y1, x2, y2, x3, y3, x, y):
                            count = count + 1
                    if count < 30:
                        ou.append(ao[i])
                    break
        return ou

    def optimize_pit_func(self):
        self.points = []
        img = cv.imread(self.img_path)
        if img.all is None:
            raise FileNotFoundError('读取失败！')
        if len(img.shape) == 3:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_gray = img
        res = cv.findContours(img_gray, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        if len(res) == 2:
            contours, _ = res
        else:
            _, contours, _ = res
        for i in range(len(contours)):
            x = []
            y = []
            tu = []
            ao = []
            hull = cv.convexHull(contours[i], returnPoints=False)
            defects = cv.convexityDefects(contours[i], hull)
            for n in range(len(hull)):
                tu.append(hull[n][0])
            try:
                for m in range(len(defects)):
                    s, e, f, d = defects[m, 0]
                    ao.append(f)
            except Exception:
                pass
            tu.sort()
            ao.sort()
            ou = self.area_optimize(contours[i], ao, tu)
            for s in ou:
                tu.append(s)
            tu.sort()
            for j in tu:
                x.append(contours[i][j][0][0])
                y.append(contours[i][j][0][1])
            # x.append(x[0])
            # y.append(y[0])
            self.points.append([x, y])
        return self.points, img_gray.shape[1]



def plt_point(points, h,name=None, scatter=False):
    """
    input: points 是一个列表，列表中是各个检测的关键点的坐标
    name : 绘图的名称
    """
    plt.figure('layer_{}'.format(name),dpi=300)
    plt.ylim(h, 0)
    plt.xlim(0,h)
    for i in range(len(points)):
        x = []
        y = []
        for j in range(len(points[i][0])):
            if scatter:
                plt.scatter(points[i][0][j], points[i][1][j])

            x.append(points[i][0][j])
            y.append(points[i][1][j])
        plt.plot(x, y)
    plt.show()

