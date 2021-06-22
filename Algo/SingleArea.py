try:
    import cv2 as cv
    import numpy as np
    import sys
    import os
    import time
    import glob
except ModuleNotFoundError as reason:
    print('错误原因是：' + str(reason))

#
# def measure_object(images, labels, n, f, savept, resolution):
#     num = 0.0
#     number = 0
#     r_num = float(resolution)
#     f.writelines('********编号{}的图片*********\n'.format(n))
#     gray = cv.cvtColor(labels, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     outimage, contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#     for i, contour in enumerate(contours):
#         x, y, z = contour.shape
#         area = cv.contourArea(contour) + ((x + 2) / 2)
#         if area > 300:
#             number += 1
#             res = cv.drawContours(images, contour, -1, (0, 0, 255), 2)
#             mm = cv.moments(contour)
#             cx = mm['m10'] / (mm['m00'] + 1)
#             cy = mm['m01'] / (mm['m00'] + 1)
#
#             im = cv.putText(images, str(number), (np.int(cx), np.int(cy)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
#             print('area{}:{}{}'.format(number, np.round(area * r_num * r_num, 3), '平方米'))
#             sys.stdout.flush()
#             f.writelines('  area{}:{}{}\n'.format(number, np.round(area * r_num * r_num, 3), '平方米'))
#             num = num + area * r_num * r_num
#
#             cv.imwrite(savept, im)
#     f.writelines('  测量面积总数值:{}平方米\n'.format(np.round(num, 3)))
#     print('测量面积总数值:{}平方米'.format(np.round(num, 3)))
#     sys.stdout.flush()
#
#
# def Contour_area(imgpath, predpath, savepath, resolu):
#     print('/************ 轮廓面积 *******************/')
#     sys.stdout.flush()
#     area_pa = os.path.dirname(os.path.dirname(savepath))
#     na = savepath.split('\\')[-1].split('.')[0]
#
#
#     with open(area_pa + '\\building_area_{}.txt'.format(na), 'w', encoding='utf-8') as f:
#         print('/************ 第1张 *******************/')
#         sys.stdout.flush()
#         pred_res = cv.imread(imgpath)
#         object_img = cv.imread(predpath)
#         n = imgpath.split(imgpath.split('.')[-1])[0].split('\\')[-1]
#         measure_object(images=pred_res, labels=object_img, n=n, f=f, savept=savepath, resolution=resolu)
#

class Contour_Detection():
    def __init__(self):
        self.detection_img_list = []
        self.pred_img_list = []

    def img_detection(self, imgs, pred_imgs, save_path, resolution):    # 检测单张或者多张图像
        root_path = os.path.dirname(save_path)
        root_file_name = time.strftime('%y_%m_%d_%H_%M_%S') + 'area.txt'
        self.detection_img_list = [i for i in imgs.split(';')[:-1]]
        self.pred_img_list = [i for i in pred_imgs.split(';')[:-1]]
        with open(os.path.join(root_path, root_file_name), 'w', encoding='utf-8') as f:
            for img_num in range(len(self.detection_img_list)):
                self.detection_func(img_path=self.detection_img_list[img_num],
                                    pred_img_path=self.pred_img_list[img_num],
                                    save_path=save_path,
                                    resolution=resolution,
                                    n=img_num,
                                    f_weight = f)


    def dir_detection(self, imgs, pred_imgs, save_path, resolution):
        root_path = os.path.dirname(save_path)
        root_file_name = time.strftime('%y_%m_%d_%H_%M_%S') + 'area.txt'
        images_path = glob.glob(imgs + '/*')
        pred_images_path = glob.glob(pred_imgs + '/*')

        with open(os.path.join(root_path, root_file_name), 'w', encoding='utf-8') as f:
            for i in range(len(images_path)):
                self.detection_func(img_path=images_path[i],
                                    pred_img_path=pred_images_path[i],
                                    save_path=save_path,
                                    resolution=resolution,
                                    n=i,
                                    f_weight=f)

    def detection_func(self, img_path, pred_img_path, save_path, resolution, n, f_weight):
        img = cv.imread(img_path)
        pred_img = cv.imread(pred_img_path)
        name = img_path.split('/')[-1].split('.')[0]
        img_name = os.path.join(save_path, name + '_area.tif')
        print('正在检测第 {} 张图像 -> {}'.format(n, img_path))
        self.measure_object(images=img, labels=pred_img, n=img_path, f=f_weight,
                            save_name=img_name, resolution=resolution)

    def measure_object(self, images, labels, n, f, save_name, resolution):
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

                im = cv.putText(images, str(number), (np.int(cx), np.int(cy)), cv.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 0), 1)
                print('area{}:{}{}'.format(number, np.round(area * r_num * r_num, 3), '平方米'))
                sys.stdout.flush()
                f.writelines('  area{}:{}{}\n'.format(number, np.round(area * r_num * r_num, 3), '平方米'))
                num = num + area * r_num * r_num

                cv.imwrite(save_name, im)
        f.writelines('  测量面积总数值:{}平方米\n'.format(np.round(num, 3)))
        print('测量面积总数值:{}平方米'.format(np.round(num, 3)))
        sys.stdout.flush()



# if __name__ == '__main__':
#     try:
#         Contour_area(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
#     except Exception as e:
#         print('错误原因是： ' + str(e))
#     # Contour_area(imgpath='D:\\demo\\dome_data\\test\\1.png',
#     #              predpath='D:\\demo\\dome_data\\pred\\1_pred.png',
#     #              savepath='D:\\demo\\dome_data\\pred\\')
