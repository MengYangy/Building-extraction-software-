try:
    import cv2 as cv
    import numpy as np
    import sys
    import os
    import glob
except ModuleNotFoundError as reason:
    print('错误原因是：' + str(reason))


def measure_object(images, labels, n, f, savept, resolution):
    num = 0.0
    number = 0
    r_num = float(resolution)
    f.writelines('********编号{}的图片*********\n'.format(n))
    gray = cv.cvtColor(labels, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    outimage, contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i, contour in enumerate(contours):
        x, y, z = contour.shape
        area = cv.contourArea(contour) + ((x + 2) / 2)
        if area > 300:
            number += 1
            res = cv.drawContours(images, contour, -1, (0, 0, 255), 2)
            mm = cv.moments(contour)
            cx = mm['m10'] / (mm['m00'] + 1)
            cy = mm['m01'] / (mm['m00'] + 1)

            im = cv.putText(images, str(number), (np.int(cx), np.int(cy)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0),
                            1)
            print('area{}:{}{}'.format(number, np.round(area * r_num * r_num, 3), '平方米'))
            sys.stdout.flush()
            f.writelines('  area{}:{}{}\n'.format(number, np.round(area * r_num * r_num, 3), '平方米'))
            num = num + area * r_num * r_num
            cv.imwrite(savept + 'img_contour_{}png'.format(n), im)
    f.writelines('测量面积总数值:{}平方米\n'.format(np.round(num, 3)))
    print('测量面积总数值:{}平方米'.format(np.round(num, 3)))
    sys.stdout.flush()


def Contour_area(imgpath, predpath, savepath, resolu):
    print('/************ 轮廓面积 *******************/')
    sys.stdout.flush()
    predimgs = glob.glob(imgpath + '*')
    labimgs = glob.glob(predpath + '*')
    area_pa = os.path.dirname(os.path.dirname(savepath))
    with open(area_pa + '\\building_areas.txt', 'w', encoding='utf-8') as f:
        for i in range(len(predimgs)):
            print('/************ 第{}张 *******************/'.format(i + 1))
            sys.stdout.flush()
            pred_res = cv.imread(predimgs[i])
            object_img = cv.imread(labimgs[i])
            n = os.listdir(imgpath)[i].split(os.listdir(imgpath)[i].split('.')[-1])[0]
            measure_object(images=pred_res, labels=object_img, n=n, f=f, savept=savepath, resolution=resolu)


if __name__ == '__main__':
    try:
        Contour_area(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except Exception as e:
        print('错误原因是： ' + str(e))
    # Contour_area(imgpath='D:\\demo\\dome_data\\test\\',
    #              predpath='D:\\demo\\dome_data\\pred\\',
    #              savepath='D:\\demo\\dome_data\\model\\')
