# -*- coding:UTF-8 -*-
"""
文件说明：
    测试文件
"""
from findPoints import FindPoints, plt_point
from vectorization import create_vector
import anglePoint
from edge_3 import detection
from utils import dataPro
import sys
import glob


points, h = detection(r'E:\data\model\other_data\tets\labels\2_127.tif')
print(h)
plt_point(points, h, name='angle',scatter=True)


def fina_vector(in_path, out_path, mode=0):
    """
    :param in_path: 输入图像
    :param out_path: 结果保存路径
    :param mode: 默认模式0 单张， 为1是 多张
    """
    points, h = detection(in_path)
    create_vector(points, h, shp_path=out_path)


if __name__ == '__main__':

    try:
        if int(sys.argv[3]) == 0:
            print('单张矢量化操作： ')
            fina_vector(sys.argv[1], sys.argv[2])
        elif int(sys.argv[3]) == 1:

            img_paths = glob.glob(sys.argv[1])
            print('多张矢量化操作： 此次矢量化共 {}张图像'.format(len(img_paths)))
            for i in img_paths:
                name = sys.argv[2] + '\\' + i.split('\\')[-1].split('.')[0] + '.shp'
                fina_vector(i, name)
                print("完成 {} 的矢量化".format(name))



    except Exception as e:
        print("错误原因是： " + str(e))