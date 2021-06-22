import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


images_path=glob.glob(r'C:\Users\hp\Documents\Tencent Files\1505257290\FileRecv\images\*.tif')
label_path=glob.glob(r'C:\Users\hp\Documents\Tencent Files\1505257290\FileRecv\pred\*.png')
result_save_path= "C:/Users/hp/Desktop/save"
print('当前路径下待检测样本数为：{}'.format(len(images_path)))


def make_path(p_path,images_name):
    flag=os.path.exists(p_path)
    if not flag:
        os.makedirs(p_path)
    c_path=p_path+'/'+images_name
    c_flag=os.path.exists(c_path)
    if not c_flag:
        os.makedirs(c_path)


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
    req = iou > 0.5
    #   no iou>0.6
    if np.any(req):
        return np.argmax(iou)
    else:
        return None


def process_td(initial_edge, erode_edge):
    initial_bbox = []
    for i in range(len(initial_edge)):
        x, y, w, h = cv.boundingRect(initial_edge[i])
        xmax = x+w
        ymax = y+h
#         M = cv.moments(initial_edge[i])
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
        initial_bbox.append([x, y, xmax, ymax, i])
    erode_bbox=[]
    for j in range(len(erode_edge)):
        x, y, w, h = cv.boundingRect(erode_edge[j])
        xmax = x+w
        ymax = y+h
#         M = cv.moments(erode_edge[j])
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
        erode_bbox.append([x, y, xmax, ymax, j])
#     find_position
#     for i in range(len(initial_bbox)):
#         print(initial_bbox[i],erode_bbox[i])
    ini_map = []
    add_map = []
    for i in range(len(initial_edge)):
        res=iou(initial_bbox[i], erode_bbox)
        if res is None:
            ini_map.append(i)
        else:
            add_map.append(res)
#     消失的与新增的
#     print(ini_map,add_map)
    disapper = []
#     print('无法对应的轮廓数有:{}个'.format(len(ini_map)))
    for i in range(len(ini_map)):
        disapper.append(initial_bbox[ini_map[i]])
#         消失的,里面的内容是:xmin,ymin,xmax,ymax,cnt_idx
    add = []
#     print('新增的区域有：{}'.format(len(erode_bbox)-len(add_map)))
    for i in range(len(erode_edge)):
        if i in add_map:
            continue
        add.append(erode_bbox[i])
#         新增的
    return disapper, add


def process_rl(initial_edge, erode_edge):
    initial_bbox = []
    for j in range(len(initial_edge)):
        if initial_edge[j] is None:
            #             消失的轮廓，其对应的坐标值以0初始化
            initial_bbox.append([0, 0, 0, 0, j])
            continue
        x, y, w, h = cv.boundingRect(initial_edge[j])
        initial_bbox.append([x, y, x + w, y + h, j])
    erode_bbox = []
    for j in range(len(erode_edge)):
        x, y, w, h = cv.boundingRect(erode_edge[j])
        erode_bbox.append([x, y, x + w, y + h, j])
    in_erode = []
    not_in_erode = []
    for i in range(len(initial_bbox)):
        res = iou(initial_bbox[i], erode_bbox)
        if res is None:
            not_in_erode.append(initial_bbox[i][4])
            continue
        in_erode.append(res)
    disapper_bbox = []
    #     1.无法对应的情况：一分为二，腐蚀加填充，本身已经是忽略的框
    for i in range(len(not_in_erode)):
        disapper_bbox.append(initial_bbox[not_in_erode[i]])
    #   新增的轮廓

    new_bbox = []
    for i in range(len(erode_edge)):
        if i in in_erode:
            continue
        new_bbox.append(erode_bbox[i])
    #         xmin,ymin,xmax,ymax,index
    return disapper_bbox, new_bbox


def erode_images_process(erode_img, contours):
    bad_erode = []
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < 100:

            cv.drawContours(erode_img, contours, i, 0, cv.FILLED)
            if area <= 10:
                continue
            else:
                print('腐蚀之后有小区快产生')
                bad_erode.append(contours[i])
    #             处理腐蚀后多余的小轮廓
    #             对处理后的图片再次进行轮廓检测
    erode_opt = erode_img[:, :, 0].copy()
    res = cv.findContours(erode_opt, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res) == 2:
        contour, _ = res
    else:
        _, contour,_ = res
    return erode_img, contour,  bad_erode


def plot_bad_erode(images, cnt):
    for i in range(len(cnt)):
        x, y, w, h = cv.boundingRect(cnt[i])
#         cv.rectangle(images,(x,y),(x+w,y+h),(0,0,255),10)
        cv.circle(images, (int(x+w/2), int(y+h/2)), 20, (0, 0, 255))


def plot_bad_erode1(images,cnt):
    for i in range(len(cnt)):
        x, y, w, h = cv.boundingRect(cnt[i])
#         cv.rectangle(images,(x,y),(x+w,y+h),(0,0,255),10)
        cv.circle(images, (int(x+w/2), int(y+h/2)), 20, (0, 255, 0))


def expand_edge(images_w,images_h,contours,idx,kernel,iter_time):
    print('膨胀阶段:面积与轮廓的补偿')
    cur_img=np.zeros((images_w,images_h),dtype=np.uint8)
    cv.drawContours(cur_img,contours,idx,255,cv.FILLED)
    erosion1 = cv.dilate(cur_img,kernel,iterations=iter_time)
    res = cv.findContours(erosion1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
    if len(res)==2:
        contours,_=res
    else:
        _,contours,_=res
    # plt.imshow(abs(erosion1-cur_img))
    # plt.show()
    return contours[0]


def detction_overlap_building(input_img, input_edge, kernel_size, iteration):
    w,h=input_img.shape[:2]
    img = input_img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    data = img.copy()
    #     print(data.shape)
    res_ini = cv.findContours(data, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res_ini) == 2:
        res1, _ = res_ini
    else:
        _, res1, _ = res_ini

    target_num = len(res1)
    #     kerne= np.ones((kernel_size,kernel_size),np.uint8)
    #     # left-right and top-down
    #     erosion = cv.erode(img,kernel,iterations=iteration)
    #     contours,_=cv.findContours(erosion,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
    #     double=len(contours)
    #     暂且只考虑单方向的腐蚀操作

    img1 = input_img.copy()
    kernel = np.ones((1, kernel_size), np.uint8)
    # top-down
    erosion1 = cv.erode(img1, kernel, iterations=iteration)
    #     print(erosion1.shape)
    gray_e1 = cv.cvtColor(erosion1, cv.COLOR_BGR2GRAY)
    #     print(gray_e1.shape)
    res2 = cv.findContours(gray_e1, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res2) == 2:
        contours1, _ = res2
    else:
        _, contours1, _ = res2
    opt_ero, contours1, bad_erode1 = erode_images_process(erosion1, contours1)
    if bad_erode1 != []:
        plot_bad_erode(erosion1, bad_erode1)
    single_td = len(contours1)

    img2 = input_img.copy()
    kernel = np.ones((kernel_size, 1), np.uint8)
    erosion2 = cv.erode(img2, kernel, iterations=iteration)
    #     对其进行了腐蚀之后，对于检测效果较差之处
    gray_e2 = cv.cvtColor(erosion2, cv.COLOR_BGR2GRAY)
    res3 = cv.findContours(gray_e2, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    if len(res3) == 2:
        contours2, _ = res3
    else:
        _, contours2, _ = res3
    opt_ero1, contours2, bad_erode2 = erode_images_process(erosion2, contours2)
    if bad_erode2 != []:
        plot_bad_erode1(erosion2, bad_erode2)
    single_rl = len(contours2)

    if (single_td == target_num) and (single_rl == target_num):
        print("没有边角重叠在一起的建筑物")
        dis = None
        add = None
        dis1 = None
        add1 = None
    else:
        if single_td != target_num:
            #             print('竖直方向产生了重叠')
            #             print("原始的轮廓个数为：{}".format(len(res1)))
            dis, add = process_td(res1, contours1)
        #             返回无法对应的轮廓索引与新增的轮廓索引
        #             bad_num=len(dis)
        #             if dis !=[]:
        #                 for i in range(len(dis)):
        #                     res1[dis[i][4]]=None
        #                     不可以仅仅用pop操作，pop一个之后，全局索引已经发生了变化
        #             print("删除冗余之后轮廓个数为：{}".format(len(res1)-bad_num))
        #             k=0
        #             if add!=[]:
        #                 for i in range(len(add)):
        #                     cur_area=(add[i][2]-add[i][0])*(add[i][3]-add[i][1])
        #                     if cur_area<=50:
        # #                         新增模块面积小于50时将其舍去
        #                         continue
        #                     res1.append(contours1[add[i][4]])
        #                     k+=1
        #             print("新增重叠目标之后轮廓个数为：{}".format(len(res1)-bad_num))
        else:
            dis = None
            add = None
        #             k=0
        #             bad_num=0

        if single_rl != target_num:
            #             print(' 水平方向产生了重叠')
            #             print("经过垂直方向腐蚀后轮廓的个数为:{}".format(len(res1)-bad_num))
            dis1, add1 = process_rl(res1, contours2)
        #             bad1_num=len(dis1)
        #             注意此处到底需不需要设置None
        #             for i in range(bad1_num):
        #                 res1[dis1[i][4]]=None

        #             print("删除水平重叠之后轮廓个数为:{}".format(len(res1)-bad1_num))
        #             k1=0
        #             for i in range(len(add1)):
        #                 if (add1[i][2]-add1[i][0])*(add1[i][3]-add1[i][0])<=50:
        #                     continue
        #                 res1.append(contours2[add1[i][4]])
        #                 k1+=0
        #             print("新增水平方向新增的轮廓个数为:{}".format(len(res1)-bad1_num))
        else:
            dis1 = None
            add1 = None

        if dis != None:
            for i in range(len(dis)):
                res1[dis[i][4]] = None

        if dis1 != None:
            for i in range(len(dis1)):
                res1[dis1[i][4]] = None

        if add != None and add1 != None:
            add_2 = []
            #             print("*********")
            #             print(len(add1))
            if len(add) >= 1 and len(add1) >= 1:
                for i in range(len(add)):
                    iou1 = iou(add[i], add1)
                    con = expand_edge(w, h, contours1, add[i][4], kernel, iteration)
                    res1.append(con)
                    if iou1 is None:
                        continue
                    add_2.append(iou1)
                    #   返回的是add1中的第几个有重叠
                for i in range(len(add1)):
                    if i in add_2:
                        continue
                    con1 = expand_edge(w, h, contours2, add1[i][4], kernel1, iteration)
                    res1.append(con1)
            elif len(add) >= 1:
                for i in range(len(add)):
                    con = expand_edge(w, h, contours1, add[i][4], kernel, iteration)
                    res1.append(con)
            else:
                for i in range(len(add1)):
                    con1 = expand_edge(w, h, contours2, add1[i][4], kernel1, iteration)
                    res1.append(con1)

        elif add != None:
            for i in range(len(add)):
                con = expand_edge(w, h, contours1, add[i][4], kernel, iteration)
                res1.append(con)
        else:
            for i in range(len(add1)):
                con1 = expand_edge(w, h, contours2, add1[i][4], kernel1, iteration)
                res1.append(con1)
            #       在不用的图上绘制出无法对应的与新增的轮廓的外接矩形框
    return res1, erosion1, erosion2, dis, add, dis1, add1


def small_target(input_img, edge, epsilon):
    approx = cv.approxPolyDP(edge, epsilon, True)
    points = approx.reshape((-1, 2))
    count = 0
    rate = 0.002
    while len(points) != 4:
        epsilon = rate * cv.arcLength(edge, True)
        rate = rate+0.002
        approx = cv.approxPolyDP(edge, epsilon,True)
        points = approx.reshape((-1, 2))
        count += 1
        if count > 10:
            break
    if len(points) == 4:
        print("小目标的优化结果为4边形")
    else:
        print("小目标的优化方法为外接最小矩形")
        rect = cv.minAreaRect(edge)
        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        points = cv.boxPoints(rect)
    return points


def big_building(img, edge, epsilon):
    epsilon = 0.005 * cv.arcLength(edge, True)
    approx = cv.approxPolyDP(edge, epsilon, True)
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


def detection(images_path, label_path, save_path="C:/Users/hp/Desktop/save"):
    img_names = images_path.split('\\')[-1].split('.')[0]
    make_path(save_path, img_names)
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

    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        cv.fillPoly(initial_img, [contours[i]], (255, 255, 255))
        if area <= 100:
            print('正在填充面积小于100的区域')
            cv.drawContours(initial_img, contours, i, 0, cv.FILLED)
            continue

    cv.imwrite(save_path + '/' + img_names + '/{}_fill_img.png'.format(img_names), initial_img)

    re, erode1, erode2, dis, add, dis1, add1 = detction_overlap_building(initial_img, contours, 7, 1)

    # 图片的保存
    chang_bbox = initial_img.copy()
    if dis != None:
        for i in range(len(dis)):
            cv.rectangle(chang_bbox, (dis[i][0], dis[i][1]), (dis[i][2], dis[i][3]), (0, 255, 0), 2)
    if dis1 != None:
        for i in range(len(dis1)):
            cv.rectangle(chang_bbox, (dis1[i][0], dis1[i][1]), (dis1[i][2], dis1[i][3]), (0, 255, 0), 2)
    cv.imwrite(save_path + '/' + img_names + "/{}_change.png".format(img_names), chang_bbox)
    #     原始的框无法对应腐蚀之后的那些框
    #     无法对应有两种可能：1.物体一分为二，导致bbox无法与之对应,对应大框
    #                         2.小目标经过腐蚀之后面积小于阈值直接被填充了，对应小框
    cv.imwrite(save_path + '/' + img_names + "/{}_erode1.png".format(img_names), erode1)
    #     标记出面积太小被填充的区域
    cv.imwrite(save_path + '/' + img_names + "/{}_erode2.png".format(img_names), erode2)
    #     与erode1同理
    if add1 is not None:
        for i in range(len(add1)):
            cv.rectangle(erode2, (add1[i][0], add1[i][1]), (add1[i][2], add1[i][3]), (0, 255, 0), 2)
    cv.imwrite(save_path + '/' + img_names + "/{}_erode_2_bbox.png".format(img_names), erode2)
    if add is not None:
        for i in range(len(add)):
            cv.rectangle(erode1, (add[i][0], add[i][1]), (add[i][2], add[i][3]), (0, 0, 255), 2)
    cv.imwrite(save_path + '/' + img_names + "/{}_erode_1_bbox.png".format(img_names), erode1)

    input_img = cv.imread(images_path)
    initial_img = input_img.copy()

    area_result = []
    contours = re
    all_coner_x = []
    all_coner_y = []
    all_coner_angel = []
    index = 0
    for i in range(len(contours)):
        if contours[i] is None:
            #             print('*********')
            continue
        #         cv.drawContours(initial_img, contours[i], -1, (0, 0, 255), 1)
        area = cv.contourArea(contours[i])
        area_result.append(int(area))
        epsilon = 0.01 * cv.arcLength(contours[i], True)

        M = cv.moments(contours[i])
        if M["m00"] <= 10:
            print('填充之后再次出现面小于10的区域')
            continue
        #         cX= int(M["m10"] / M["m00"])
        #         cY= int(M["m01"]/ M["m00"])

        if area < 150:
            points = small_target(initial_img, contours[i], epsilon=epsilon)
        elif 150 < area < 300:
            epsilon = 5 * epsilon
            approx = cv.approxPolyDP(contours[i], epsilon, True)
            points = approx.reshape((-1, 2))
        elif 3000 < area < 8000:
            points = big_building(initial_img, contours[i], epsilon=epsilon)
        elif 8000 < area <= 15000:
            points = big_building1(initial_img, contours[i], epsilon=epsilon)
        elif area > 15000:
            points = big_building2(initial_img, contours[i], epsilon=epsilon)
        else:
            approx = cv.approxPolyDP(contours[i], epsilon, True)
            points = approx.reshape((-1, 2))

        # all_coner.append(points)
        for j in range(len(points)):
            cv.circle(initial_img, tuple(points[j]), 2, (0, 255, 0))
        _, _, angel = cv.minAreaRect(contours[i])
        x1 = points[:, 0]
        x1 = list(x1)
        x1.append(points[0, 0])
        y1 = points[:, 1]
        y1 = list(y1)
        y1.append(points[0, 1])
        plt.plot(x1, y1)

        all_coner_x.append(x1)
        all_coner_y.append(y1)
        all_coner_angel.append(angel)

    plt.imshow(initial_img, interpolation='bicubic')
    plt.savefig(save_path + '/' + img_names + '/{}_optimizers.jpg'.format(img_names), dpi=300)
    # plt.show()
    plt.cla()
    return all_coner_x, all_coner_y, img.shape[0],all_coner_angel


all_images_x = []
all_images_y = []
all_images_shape = []
all_images_angel=[]
for i in range(len(images_path)):
    x, y, shape, angel = detection(images_path=images_path[i], label_path=label_path[i], save_path=result_save_path)
    all_images_x.append(x)
    all_images_y.append(y)
    all_images_shape.append(shape)
    all_images_angel.append(angel)

