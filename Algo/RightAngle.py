# -*- coding:UTF-8 -*-
"""
文件说明：
    直角约束
"""

try:
    import cv2 as cv
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    import anglePoint
    import glob
    from findPoints import plt_point
except Exception as e:
    print("错误原因是： " + str(e))


def insert_point(x1, y1, x2, y2, k1, k2):
    if k1 * k2 < 0 and k1!=0 and k2!=0 and k1 != np.inf and k2 != np.inf:
        b1 = y1 - k1 * x1
        b2 = y2 - k2 * x2
        xm = (b2 - b1) / (k1 - k2)
        ym = k1 * xm + b1
        return int(xm + 0.5), int(ym + 0.5)
    elif k1 * k2 > 0 and k1!=0 and k2!=0:
        k1 = -(1 / k1)
        b1 = y1 - k1 * x1
        b2 = y2 - k2 * x2
        xm = (b2 - b1) / (k1 - k2)
        ym = k1 * xm + b1
        return int(xm + 0.5), int(ym + 0.5)
    elif k1 ==0 or k2 == np.inf:
        return x2,y1
    elif k2 ==0 or k1 == np.inf:
        return x1,y2

def right_Angle(points, angles):
    all_xy = []

    for j in range(len(points)):
        print('\n正在处理第{}个轮廓，其主方向为{}'.format(j + 1, angles[j]))
        x = []  # x坐标
        y = []  # y坐标
        k = []  # 斜率
        angle = []  # 角度
        temp_points = points[j].copy()
        if abs(angles[j]) == 0 or abs(angles[j]) == 90:
            print(j)
            print([points[j][0], points[j][1]])
            all_xy.append([points[j][0], points[j][1]])
            #         print(temp_points)
            print(len(points[j][0]), len(points[j][1]))
            continue


        elif points[j][0].count(0) > 0 or points[j][1].count(0) > 0 or points[j][0].count(512) > 0 or points[j][
            1].count(512) > 0:
            print('******************************************************************************************', j)
            all_xy.append([points[j][0], points[j][1]])
            continue
        else:
            for i in range(len(points[j][0]) - 1):
                x1 = points[j][0][i]
                x2 = points[j][0][i + 1]

                y1 = points[j][1][i]
                y2 = points[j][1][i + 1]
                if x1 == x2:
                    k.append(np.inf)
                    angle.append(90)
                else:
                    k.append((y2 - y1) / (x2 - x1))
                    angle.append(math.degrees(math.atan((y2 - y1) / (x2 - x1))))
            for i in range(len(points[j][0]) - 1):
                x1 = points[j][0][i]
                x2 = points[j][0][i + 1]

                y1 = points[j][1][i]
                y2 = points[j][1][i + 1]

                temp_angle = angle[i]  # 每一条边的角度

                if (angles[j] - 10 < temp_angle < angles[j] + 10) or (
                        angles[j] + 80 < temp_angle < angles[j] + 100):  # 判断角度阈值，在阈值内部，不处理，
                    x.append(x1)
                    y.append(y1)
                    print('不需优化目标的第{}条边，坐标{}:{}->{}:{},角度为:{}'.format(i + 1, x1, y1, x2, y2, temp_angle))
                else:
                    print('优化目标的第{}条边，坐标{}:{}->{}:{},角度为:{}'.format(i + 1, x1, y1, x2, y2, temp_angle))
                    x.append(x1)
                    y.append(y1)

                    if i == 0:
                        pro = -1
                        sub = i + 1
                    elif i == len(points[j][0]) - 2:
                        pro = i - 1
                        sub = 0
                    else:
                        pro = i - 1
                        sub = i + 1
                    print('pro={}, sub={}'.format(pro, sub))
                    print('k1={}, k2={}'.format(k[pro], k[sub]))
                    xm, ym = insert_point(x1, y1, x2, y2, k[pro], k[sub])
                    print('优化方式，插入坐标{}:{}'.format(xm, ym))
                    x.append(xm)
                    y.append(ym)

            x.append(x[0])
            y.append(y[0])
            print('k= ', k)
        all_xy.append([x, y])

    return all_xy


class Right_Angle(object):
    def __init__(self):
        pass

    def get_angle(self, points):
        """
                input:某一个建筑物的轮廓点坐标
                return: 该建筑物每一条边的角度angle,和斜率K
        """
        angle = []
        k = []
        for i in range(len(points[0]) - 1):
            x1 = points[0][i]
            y1 = points[1][i]
            x2 = points[0][i + 1]
            y2 = points[1][i + 1]
            if x2 == x1:
                angle.append(90.0)
                k.append(np.inf)
            else:
                angle.append(math.degrees(math.atan((y2 - y1) / (x2 - x1))))
                k.append((y2 - y1) / (x2 - x1))
        return angle, k

    def optimizationFunc(self,points, angles, angleThreshold=8):
        point_xy = []
        for i in range(len(points)):  # 遍历每一个建筑物
            print("第{}个建筑物的主方向角度为{}".format(i + 1, angles[i]))
            angle, k = self.get_angle(points[i])
            x = []
            y = []
            x.append(points[i][0][0])
            y.append(points[i][1][0])

            for j in range(len(angle)):  # 遍历每一条边
                x1 = points[i][0][j]
                y1 = points[i][1][j]
                x2 = points[i][0][j + 1]
                y2 = points[i][1][j + 1]
                add_station = True
                #             if angles[i]%90-10 < angle[j]%90 < angles[i]%90+10: # 条件的优化，把第1、2、3、4象限的情况统一到一个象限内处理
                # angles[i]-10 < angle[j] < angles[i]+10 or angles[i]-10 < angle[j]%90 < angles[i]+10
                if (angles[i] - angleThreshold < angle[j] < angles[i] + angleThreshold
                        or angles[i] + 90 - angleThreshold < angle[j] < angles[i] + 90 + angleThreshold
                        or angles[i] - 90 + angleThreshold > angle[j] > angles[i] - 90 - angleThreshold
                        or angles[i] + 180 + angleThreshold > angle[j] > angles[i] + 180 - angleThreshold):
                    print("第{}个建筑物的第{}条边，坐标为{}:{} -> {}:{},角度为{}, 不需要优化".format(i + 1, j + 1, x1, y1, x2, y2, angle[j]))
                    x.append(x2)
                    y.append(y2)
                else:
                    try:
                        print("第{}个建筑物的第{}条边，坐标为{}:{} -> {}:{},角度为{}, 需要优化".format(i + 1, j + 1, x1, y1, x2, y2,
                                                                                   angle[j]))
                        '''
                        先判断当前边的后一条边是否需要优化
                        '''
                        if j == 0:
                            front = -1
                            back = j + 1
                        elif j == len(angle) - 1:
                            front = j - 1
                            back = 0
                        else:
                            front = j - 1
                            back = j + 1
                        # angles[i]-10 < angle[back] < angles[i]+10 or angles[i]-10 < angle[back]%90 < angles[i]+10
                        #                 if angles[i]%90-10 < angle[back]%90 < angles[i]%90+10:
                        if (angles[i] - angleThreshold < angle[back] < angles[i] + angleThreshold
                                or angles[i] - 90 + angleThreshold > angle[back] > angles[i] - 90 - angleThreshold
                                or angles[i] + 90 - angleThreshold < angle[back] < angles[i] + 90 + angleThreshold
                                or angles[i] + 180 - angleThreshold < angle[back] < angles[i] + 180 + angleThreshold):
                            '''
                            如果后一条边不需要优化，此时判断 front边和back边的相对位置
                            '''
                            print("\t优化方式：只需优化第 {} 边".format(j + 1))
                            if abs(abs(angle[front] - angle[back]) - 90) < 10:
                                '''两条边是垂直状态'''
                                if k[front] == np.inf:
                                    '''如果front边是90°'''
                                    #                             print('\t\tfront边是90°')
                                    xm = x1
                                    ym = k[back] * (xm - x2) + y2
                                    print('\t\t两边相互垂直边，front边是90°，插入坐标{}:{}'.format(xm, ym))
                                elif k[back] == np.inf:
                                    '''如果back边是90°'''

                                    xm = x2
                                    ym = k[front] * (xm - x1) + y1
                                    print('\t\t两边相互垂直边，back边是90°，插入坐标{}:{}'.format(xm, ym))
                                else:
                                    '''两边都不是90°,
                                    需要判断该交点是在两点之间，还是在两点的延长线
                                    '''

                                    b1 = y1 - k[front] * x1
                                    b2 = y2 - k[back] * x2
                                    xm = (b2 - b1) / (k[front] - k[back])
                                    ym = k[front] * xm + b1

                                    x3 = points[i][0][back + 1]
                                    y3 = points[i][1][back + 1]
                                    d1 = (x3 - x2) ** 2 + (y3 - y2) ** 2
                                    d2 = (x3 - xm) ** 2 + (y3 - ym) ** 2
                                    if d1 > d2:
                                        add_station = False  # 为False 则不添加xm,ym
                                        print(
                                            '\t\t两边相互垂直边都不是90°，add_station = False,不插入坐标{}:{}, d1为{}  d2为{}'.format(xm,
                                                                                                                    ym,
                                                                                                                    d1,
                                                                                                                    d2))
                                    else:
                                        print('\t\t两边相互垂直边都不是90°，插入坐标{}:{}, d1为{}  d2为{}'.format(xm, ym, d1, d2))

                            elif abs(angle[front] - angle[back]) < 10 or 170 < abs(angle[front] - angle[back]) < 190:
                                '''如果两边是平行状态,求front与back的法线的交点'''
                                print('\t两边是平行状态,求front与back的法线的交点')
                                if k[front] == np.inf:
                                    xm = x1
                                    ym = y2
                                    print('\t\tfront边是90°，插入坐标{}:{}'.format(xm, ym))
                                elif k[back] == np.inf:
                                    xm = x2
                                    ym = y1
                                    print('\t\tback边是90°，插入坐标{}:{}'.format(xm, ym))
                                elif k[front] == 0 or k[back] == 0:
                                    xm, ym = x2, y1
                                else:
                                    b1 = y1 - k[front] * x1
                                    b2 = y2 + 1 / k[back] * x2
                                    xm = (b2 - b1) / (k[front] + 1 / k[back])
                                    ym = k[front] * xm + b1
                                    print('xm = {}, ym = {}'.format(xm, ym))
                                    print('\t\t两个平行边都不是90°，插入坐标{}:{}'.format(xm, ym))

                            else:
                                '''其他状态'''
                                if (angle[back] - angle[front]) % 90 > 20:
                                    if k[back] == np.inf:
                                        #                                 xm = x2
                                        #                                 ym = y1 + k[back]*(xm-x1)
                                        ym = y2
                                        xm = x1
                                    elif k[back] == 0:
                                        xm = x2
                                        ym = y1 + k[back] * (xm - x1)
                                    else:
                                        b2 = y2 + 1 / k[back] * x2
                                        b1 = y1 - k[back] * x1
                                        xm = (b2 - b1) / (1 / k[back] + k[back])
                                        ym = k[back] * xm + b1


                                else:

                                    b1 = y1 - k[front] * x1
                                    b2 = y2 - k[back] * x2
                                    xm = (b2 - b1) / (k[front] - k[back])
                                    ym = k[front] * xm + b1

                                    print('k[front]={}:{}  k[back]={}:{}'.format(k[front], angle[front], k[back],
                                                                                 angle[back]))
                                print('\t\t两个边不相互平行，也不相互垂直，插入坐标{}:{}'.format(xm, ym))

                    except RuntimeWarning as e:
                        add_station = False
                        print(
                            "************************************************************************出现错误，未完成坐标点的插入，错误原因是" + str(
                                e))

                        d1 = (x1 - x2) ** 2 + (y1 - y2) ** 2  # 原始待优化边的长度
                        d2 = (x2 - xm) ** 2 + (y2 - ym) ** 2  # 优化后边的长度

                        #                     if d2 < d1/4 or d2 > 2*d1:
                        if d2 > 2 * d1:
                            add_station = False  # 为False 则不添加xm,ym
                            print('\t\td1={}, d2={},add_station = False,不插入坐标{}:{}, '.format(d1, d2, xm, ym))

                        if xm > 512:
                            xm = 512
                        if xm < 0:
                            xm = 0
                        if ym > 512:
                            ym = 512
                        if ym < 0:
                            ym = 0
                        if 0 <= xm <= 512 and 0 <= ym <= 512:
                            pass
                        else:
                            add_station = False
                        if add_station:
                            x.append(int(xm + 0.5))
                            y.append(int(ym + 0.5))
                        x.append(x2)
                        y.append(y2)


                    else:
                        '''
                        如果后一条边需要优化
                        '''
                        x.append(x2)
                        y.append(y2)
                        print('\t\t需要连续优化第{}后的多条边，插入坐标{}:{}'.format(j + 1, x2, y2))
                        print(' NOT Processed!')
            point_xy.append([x, y])
        return point_xy



if __name__ == "__main__":
    label_path = glob.glob(r'E:\data\model\other_data\tets\labels\*.tif')
    # label_path = glob.glob(r'E:\data\model\other_data\ChainNet\Den_add\res\pred\*')
    # print('当前路径下待检测样本数为：{}'.format(len(label_path)))
    right_angle = Right_Angle()
    for i in range(5):
        points, shape,angles = anglePoint.detection(label_path[i])
        print(len(angles),len(points))
        # all_points = right_Angle(points, angles)
        all_points = right_angle.optimizationFunc(points, angles)
        plt_point(all_points, shape)
        print(all_points)