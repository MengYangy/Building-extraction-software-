import time
import cv2 as cv
# import tensorflow as tf
import sys



def myPrint(p):
    # img = cv.imread(r'D:\ttt\data\labs\0.png')
    # w,h,_ = img.shape

    # for i in range(w):
    #     for j in range(h):
    #         time.sleep(0.5)
    #         print(img[i,j,:])
    #         a = tf.math.add(img[i, j, :], img[i, j, :])
    #         print(a)
    #         cv.imwrite('./img/{}_{}.png'.format(i,j), img)
    # print('\033[0;31;40m')
    for i in range(10+int(p)):
        print(i)
        time.sleep(1)


if __name__ == '__main__':
    myPrint(sys.argv[1])