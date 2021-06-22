import cv2 as cv
import numpy as np
import os
import glob
from findPoints import plt_point


"""
    算法功能： detection( label_path)
            优化建筑物边缘。
    input:
            二值化图像路径
    return: all_coner, img.shape[0], all_coner_angel
            优化后的建筑物角点，输入图像尺寸h，每个建筑物的主方向      
"""


def make_path(p_path,images_name):
    flag=os.path.exists(p_path)
    if not flag:
        os.makedirs(p_path)
    c_path=p_path+'/'+images_name
    c_flag=os.path.exists(c_path)
    if not c_flag:
        os.makedirs(c_path)


def dilate_process(h,w,contours,kernel,iter_time):
    result=[]
    for j in range(len(contours)):
        cur_img=np.zeros((h,w),dtype=np.uint8)
        # print(len(contours))
        cv.drawContours(cur_img,contours,j,255,cv.FILLED)
        dilate1 = cv.dilate(cur_img,kernel,iterations=iter_time)
        res = cv.findContours(dilate1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
        if len(res)==2:
            contours1,_=res
        else:
            _,contours1,_=res
        result.append(contours1[0])
    print('膨胀阶段')
    return result


def fill_small_target(img,contours):
    fill_flag=False
    for i in range(len(contours)):
        area=cv.contourArea(contours[i])
        cv.fillPoly(img,[contours[i]],(255,255,255))
        if area<=80:
            fill_flag = True
            print('正在填充腐蚀之后面积小于80的区域')
            cv.drawContours(img,contours,i,0,cv.FILLED)
            continue
    return img,fill_flag


def erode_process(img,kernel_size,iteration):
    erode=img.copy()
    kernel = np.ones((1,kernel_size),np.uint8)
    erosion1 = cv.erode(erode,kernel,iterations=iteration)
    _,contours1,_ = cv.findContours(erosion1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
    if len(contours1)==1:
        print('此处没有发生水平方向的重叠')
        return None
    else:
        print('此物体发生了水平方向的重叠')
        erosion1,flag=fill_small_target(erosion1,contours1)
        if not flag:
            print('没有可以填充的小物体存在')
            h,w = img.shape
            cnt=dilate_process(h,w,contours1,kernel,iteration)
            return cnt
        else:
            _,contours1,_ = cv.findContours(erosion1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
            h,w = img.shape
            cnt=dilate_process(h,w,contours1,kernel,iteration)
            return cnt


def erode_process1(img,kernel_size,iteration):
    erode=img.copy()
    kernel = np.ones((kernel_size,1),np.uint8)
    erosion1 = cv.erode(erode,kernel,iterations=iteration)
    _,contours1,_ = cv.findContours(erosion1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
    if len(contours1)==1:
        print('此处没有发生竖直方向的重叠')
        return None
    else:
        print('此物体发生了竖直方向的重叠')
        erosion1,flag=fill_small_target(erosion1,contours1)
        if not flag:
            print('没有可以填充的小物体存在')
            h,w = img.shape
            cnt=dilate_process(h,w,contours1,kernel,iteration)
            return cnt
        else:
            _,contours1,_ = cv.findContours(erosion1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
            h,w = img.shape
            cnt=dilate_process(h,w,contours1,kernel,iteration)
            return cnt


def iou(initial_bbox, erode_bbox):
    initial_bbox = np.array(initial_bbox)
    erode_bbox = np.array(erode_bbox)
    #     print(initial_bbox[:4])

    inter_left = np.maximum(initial_bbox[:2], erode_bbox[:, :2])
    inter_right = np.minimum(initial_bbox[2:4], erode_bbox[:, 2:4])
    inter_wh = inter_right - inter_left
    inter_wh = np.maximum(inter_wh, 0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    #     print(inter_area)
    ini_area = (initial_bbox[2] - initial_bbox[0]) * (initial_bbox[3] - initial_bbox[1])
    prior_area = (erode_bbox[:, 2] - erode_bbox[:, 0]) * (erode_bbox[:, 3] - erode_bbox[:, 1])
    union_area = ini_area + prior_area - inter_area
    iou = inter_area / union_area
    #     print(iou)
    req = iou > 0.7
    #   no iou>0.6
    if np.any(req):
        return np.argmax(iou)
    else:
        return None


def cnt_bbox(cnt1, cnt2):
    bbox1 = []
    bbox2 = []
    all_cnt = []
    match_idx = []
    for i in range(len(cnt1)):
        x, y, w, h = cv.boundingRect(cnt1[i])
        bbox1.append([x, y, x + w, y + h, i])
    for i in range(len(cnt2)):
        x, y, w, h = cv.boundingRect(cnt2[i])
        bbox2.append([x, y, x + w, y + h, i])
    for i in range(len(bbox1)):
        cur_iou = iou(bbox1[i], bbox2)
        all_cnt.append(cnt1[i][4])
        if cur_iou is not None:
            match_idx.append(iou)
    for i in range(len(bbox2)):
        if bbox2[i][4] in match_idx:
            continue
        all_cnt.append(cnt2[bbox2[i][4]])
    return all_cnt


def small_target(input_img,edge,epsilon):
    approx = cv.approxPolyDP(edge,epsilon,True)
    points = approx.reshape((-1, 2))
    count=0
    rate=0.002
    while len(points)!=4:
        epsilon = rate * cv.arcLength(edge, True)
        rate=rate+0.002
        approx = cv.approxPolyDP(edge,epsilon,True)
        points = approx.reshape((-1, 2))
        count+=1
        if count>10:
            break
    if len(points)==4:
        print("小目标的优化结果为4边形")
    else:
        print("小目标的优化方法为外接最小矩形")
        rect = cv.minAreaRect(edge)
        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        points = cv.boxPoints(rect)
    return points


def big_building(img,edge,epsilon):
    epsilon = 0.005 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
    points = approx.reshape((-1, 2))
    return points


def big_building1(img,edge,epsilon):
    epsilon = 0.004 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
    points = approx.reshape((-1, 2))
    return points


def big_building2(img,edge,epsilon):
    epsilon = 0.002 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge,epsilon, True)
    points = approx.reshape((-1, 2))
    return points


def detection( label_path):
    img_names = label_path.split('\\')[-1].split('.')[0]

    print(img_names)

    img = cv.imread(label_path)
    #     cimg=img[:,:,0].copy()
    cimg = img.copy()
    cimg = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)
    # RGB------>Gray
    initial_img = img.copy()
    res = cv.findContours(cimg, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res) == 2:
        contours, idx = res
    else:
        _, contours, idx = res
    # print(len(contours))
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        cv.fillPoly(initial_img, [contours[i]], (255, 255, 255))
        if area <= 100:
            print('正在填充面积小于100的区域')
            cv.drawContours(initial_img, contours, i, 0, cv.FILLED)
            continue
    process_img = initial_img[:, :, 0].copy()
    res1 = cv.findContours(process_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    _,contours1, idx1 = res1
    # print(len(contours1))

    all_cnt = []
    for i in range(len(contours1)):
        plot_img = np.zeros((512, 512), dtype=np.uint8)
        cv.drawContours(plot_img, contours1, i, 255, cv.FILLED)

        cur_cnt = erode_process(plot_img, 5, 1)
        cur_cnt1 = erode_process1(plot_img, 5, 1)

        if cur_cnt is None and cur_cnt1 is None:
            all_cnt.append(contours1[i])
        elif cur_cnt is not None and cur_cnt1 is not None:
            cnt = cnt_bbox(cur_cnt, cur_cnt1)
            for j in range(len(cnt)):
                all_cnt.append(cnt[j])
        elif cur_cnt is not None:
            for j in range(len(cur_cnt)):
                all_cnt.append(cur_cnt[j])
        else:
            for j in range(len(cur_cnt1)):
                all_cnt.append(cur_cnt1[j])

    initial_img = np.ones((img.shape))*255

    area_result = []
    contours = all_cnt
    all_coner = []
    all_coner_angel = []

    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        area_result.append(int(area))
        epsilon = 0.01 * cv.arcLength(contours[i], True)
        M = cv.moments(contours[i])
        if M["m00"] <= 10:
            print('填充之后再次出现面小于10的区域')
            continue
        if area < 150:
            points = small_target(initial_img, contours[i], epsilon=epsilon)
        elif 150 < area < 300:
            epsilon = 5 * epsilon
            approx = cv.approxPolyDP(contours[i], epsilon, True)
            points = approx.reshape((-1, 2))
        elif area >= 3000 and area < 8000:
            points = big_building(initial_img, contours[i], epsilon=epsilon)
        elif 8000 < area <= 15000:
            points = big_building1(initial_img, contours[i], epsilon=epsilon)
        elif area > 15000:
            points = big_building2(initial_img, contours[i], epsilon=epsilon)
        else:
            approx = cv.approxPolyDP(contours[i], epsilon, True)
            points = approx.reshape((-1, 2))
        for j in range(len(points)):
            cv.circle(initial_img, tuple(points[j]), 2, (0, 255, 0))

        _, _, angel = cv.minAreaRect(contours[i])
        x1 = points[:, 0]
        x1 = list(x1)
        x1.append(points[0, 0])
        y1 = points[:, 1]
        y1 = list(y1)
        y1.append(points[0, 1])

        all_coner.append([x1, y1])
        all_coner_angel.append(angel)


    return all_coner, img.shape[0], all_coner_angel

if __name__ == "__main__":
    # label_path = glob.glob(r'E:\data\model\other_data\ChainNet\Den_add\res\pred\*')
    label_path = glob.glob(r'E:\data\model\other_data\tets\labels\*.tif')
    # print('当前路径下待检测样本数为：{}'.format(len(label_path)))
    all_images_x = []
    all_images_y = []
    all_images_shape = []
    for i in range(5):
        print(label_path[i])
        point, h,angles = detection(label_path[i])
        plt_point(point, h)
