# -*- coding:UTF-8 -*-
"""
文件说明：
    数据增强类
"""
try:
    import sys
    import cv2 as cv
    import numpy as np
    import random
    import tensorflow as tf
    import glob
    from tqdm import tqdm
    from tqdm._tqdm import trange
except Exception as reason:
    print('错误原因是：' + str(reason))


class DataPro(object):
    """
    示例：pro = DataPro(in_img_path=None, in_lab_path=None, out_img_path=None, out_lab_path=None,
                        img_num=None, img_w=None, img_h=None)
    功能：数据增强
          pro.crop_pro()    :   随机裁剪+镜像
    参数：
        in_img_path ： 输入图像的路径
        in_lab_path ： 输入标签的路径
        out_img_path ：保存图像路径
        out_lab_path ：保存标签路径
        img_num ：   裁剪数量
        img_w   ：   裁剪宽度
        img_h   ：   裁剪高度
    """
    def __init__(self, in_img_path, in_lab_path, out_img_path, out_lab_path, img_num, img_w, img_h):
        self.in_img_path = in_img_path
        self.in_lab_path = in_lab_path
        self.out_img_path = out_img_path
        self.out_lab_path = out_lab_path
        self.img_num = img_num
        self.img_w = img_w
        self.img_h = img_h


    def crop_pro(self):
        img_num = int(self.img_num)
        img_w = int(self.img_w)
        img_h = int(self.img_h)
        counts = 3
        count = 0
        count_s = 1
        images = glob.glob(self.in_img_path + '/*')
        labels = glob.glob(self.in_lab_path + '/*')
        for i in range(len(images)):
            image = cv.imread(images[i])
            label = cv.imread(labels[i])
            for j in range(counts):
                [weight, height, tong] = image.shape  # 获得原始图片的大小
                if j == 1:  # 数据增强，左右翻转
                    image = tf.image.flip_left_right(image).numpy()
                    label = tf.image.flip_left_right(label).numpy()
                    count_s += 1
                elif j == 2:  # 数据增强， 上下翻转
                    image = tf.image.flip_up_down(image).numpy()
                    label = tf.image.flip_up_down(label).numpy()
                    count_s += 1
                while count < img_num * count_s:
                    random_width = random.randint(0, weight - img_w - 1)
                    random_height = random.randint(0, height - img_h - 1)
                    train_roi = image[random_width:random_width + img_w, random_height:random_height + img_h]
                    label_roi = label[random_width:random_width + img_w, random_height:random_height + img_h]
                    cv.imwrite(self.out_img_path + '/%d.png' % (count), train_roi)
                    cv.imwrite(self.out_lab_path + '/%d.png' % (count), label_roi)
                    count += 1
                    print(count)
                    sys.stdout.flush()
            count_s = count_s + 1

    def img_process(self,img, lab):
        i = random.randint(0, 2)
        if i == 0:
            image = tf.image.flip_left_right(img).numpy()
            label = tf.image.flip_left_right(lab).numpy()
            return image, label
        if i == 1:
            image = tf.image.flip_up_down(img).numpy()
            label = tf.image.flip_up_down(lab).numpy()
            return image, label
        if i == 2:
            return img, lab
    def func(self):
        images = glob.glob('E:/data/model/one/data1/train/*')
        labels = glob.glob('E:/data/model/one/data1/label/*')
        images.sort()
        labels.sort()
        for i in trange(len(images) // 4):
            img1 = cv.imread(images[i])
            lab1 = cv.imread(labels[i])
            img1_1, lab1_1 = self.img_process(img1[:256, :256, :], lab1[:256, :256, :])
            img1_2, lab1_2 = self.img_process(img1[256:, 256:, :], lab1[256:, 256:, :])
            img1_3, lab1_3 = self.img_process(img1[256:, 0:256, :], lab1[256:, 0:256, :])
            img1_4, lab1_4 = self.img_process(img1[0:256, 256:, :], lab1[0:256, 256:, :])
            img1_5, lab1_5 = self.img_process(img1[128:384, 128:384, :], lab1[128:384, 128:384, :])

            img2 = cv.imread(images[i + 1184])
            lab2 = cv.imread(labels[i + 1184])
            img2_1, lab2_1 = self.img_process(img2[:256, :256, :], lab2[:256, :256, :])
            img2_2, lab2_2 = self.img_process(img2[256:512, 256:512, :], lab2[256:512, 256:512, :])
            img2_3, lab2_3 = self.img_process(img2[256:512, 0:256, :], lab2[256:512, 0:256, :])
            img2_4, lab2_4 = self.img_process(img2[0:256, 256:512, :], lab2[0:256, 256:512, :])
            img2_5, lab2_5 = self.img_process(img2[128:384, 128:384, :], lab2[128:384, 128:384, :])

            img3 = cv.imread(images[i + 2368])
            lab3 = cv.imread(labels[i + 2368])
            img3_1, lab3_1 = self.img_process(img3[:256, :256, :], lab3[:256, :256, :])
            img3_2, lab3_2 = self.img_process(img3[256:512, 256:512, :], lab3[256:512, 256:512, :])
            img3_3, lab3_3 = self.img_process(img3[256:512, 0:256, :], lab3[256:512, 0:256, :])
            img3_4, lab3_4 = self.img_process(img3[0:256, 256:512, :], lab3[0:256, 256:512, :])
            img3_5, lab3_5 = self.img_process(img3[128:384, 128:384, :], lab3[128:384, 128:384, :])

            img4 = cv.imread(images[i + 3552])
            lab4 = cv.imread(labels[i + 3552])
            img4_1, lab4_1 = self.img_process(img4[:256, :256, :], lab4[:256, :256, :])
            img4_2, lab4_2 = self.img_process(img4[256:512, 256:512, :], lab4[256:512, 256:512, :])
            img4_3, lab4_3 = self.img_process(img4[256:512, 0:256, :], lab4[256:512, 0:256, :])
            img4_4, lab4_4 = self.img_process(img4[0:256, 256:512, :], lab4[0:256, 256:512, :])
            img4_5, lab4_5 = self.img_process(img4[128:384, 128:384, :], lab4[128:384, 128:384, :])

            image1 = np.zeros([512, 512, 3])
            label1 = np.zeros([512, 512, 3])
            image1[:256, :256, :] = img1_1
            label1[:256, :256, :] = lab1_1
            image1[256:, 256:, :] = img2_1
            label1[256:, 256:, :] = lab2_1
            image1[:256, 256:, :] = img3_1
            label1[:256, 256:, :] = lab3_1
            image1[256:, :256, :] = img4_1
            label1[256:, :256, :] = lab4_1

            image2 = np.zeros([512, 512, 3])
            label2 = np.zeros([512, 512, 3])
            image2[:256, :256, :] = img1_2
            label2[:256, :256, :] = lab1_2
            image2[256:, 256:, :] = img2_2
            label2[256:, 256:, :] = lab2_2
            image2[:256, 256:, :] = img3_2
            label2[:256, 256:, :] = lab3_2
            image2[256:, :256, :] = img4_2
            label2[256:, :256, :] = lab4_2

            image3 = np.zeros([512, 512, 3])
            label3 = np.zeros([512, 512, 3])
            image3[:256, :256, :] = img1_3
            label3[:256, :256, :] = lab1_3
            image3[256:, 256:, :] = img2_3
            label3[256:, 256:, :] = lab2_3
            image3[:256, 256:, :] = img3_3
            label3[:256, 256:, :] = lab3_3
            image3[256:, :256, :] = img4_3
            label3[256:, :256, :] = lab4_3

            image4 = np.zeros([512, 512, 3])
            label4 = np.zeros([512, 512, 3])
            image4[:256, :256, :] = img1_4
            label4[:256, :256, :] = lab1_4
            image4[256:, 256:, :] = img2_4
            label4[256:, 256:, :] = lab2_4
            image4[:256, 256:, :] = img3_4
            label4[:256, 256:, :] = lab3_4
            image4[256:, :256, :] = img4_4
            label4[256:, :256, :] = lab4_4

            image5 = np.zeros([512, 512, 3])
            label5 = np.zeros([512, 512, 3])
            image5[:256, :256, :] = img1_5
            label5[:256, :256, :] = lab1_5
            image5[256:, 256:, :] = img2_5
            label5[256:, 256:, :] = lab2_5
            image5[:256, 256:, :] = img3_5
            label5[:256, 256:, :] = lab3_5
            image5[256:, :256, :] = img4_5
            label5[256:, :256, :] = lab4_5
    def copy_paste_pro(self):
        pass