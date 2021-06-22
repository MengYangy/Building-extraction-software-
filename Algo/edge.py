import numpy as np
import  cv2 as cv
import  matplotlib.pyplot as plt
from findPoints import plt_point

# image_path = input("请输入原始图片的路径:")
image_path = r'E:\work\remote\V4\Algo\img\2_102_a.tif'
# image_path1 = input("请输入二值图片的路径:")
image_path1 = r'E:\work\remote\V4\Algo\img\2_102_b.tif'
input_img = cv.imread(image_path)
img = cv.imread(image_path1)
cimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cimg = img[:, :, 0]
initial_img = img.copy()
res = cv.findContours(cimg, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
if len(res) == 2:
    contours, _ = res
else:
    _, contours, _ = res


def iou(initial_bbox, erode_bbox):
    initial_bbox = np.array(initial_bbox)
    erode_bbox = np.array(erode_bbox)
    # print(initial_bbox[:4])

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


def process_td(initial_edge,erode_edge):
    initial_bbox = []
    for i in range(len(initial_edge)):
        x, y, w, h = cv.boundingRect(initial_edge[i])
        x_max = x+w
        y_max = y+h
#         M = cv.moments(initial_edge[i])
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
        initial_bbox.append([x, y, x_max, y_max, i])
    erode_bbox = []
    for j in range(len(erode_edge)):
        x, y, w, h = cv.boundingRect(erode_edge[j])
        x_max = x+w
        y_max = y+h
#         M = cv.moments(erode_edge[j])
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
        erode_bbox.append([x, y, x_max, y_max, j])
#     find_bbox_position
#     for i in range(len(initial_bbox)):
#         print(initial_bbox[i],erode_bbox[i])
    ini_map = []
    add_map = []
    for i in range(len(initial_edge)):
        res = iou(initial_bbox[i], erode_bbox)
        if res is None:
            ini_map.append(i)
        else:
            add_map.append(res)
#     消失的与新增的
#     print(ini_map,add_map)
    disapper = []
    for i in range(len(ini_map)):
        disapper.append(initial_bbox[ini_map[i]])
#         消失的
    add = []
    for i in range(len(erode_edge)):
        if i in add_map:
            continue
        add.append(erode_bbox[i])
#         新增的
    return disapper, add


def process_rl(initial_edge,erode_edge):
    initial_bbox=[]
    for j in range(len(initial_edge)):
        if initial_edge[j] is None:
            continue
        x, y, w, h = cv.boundingRect(initial_edge[j])
        initial_bbox.append([x, y ,x+w ,y+h ,j])
    erode_bbox=[]
    for j in range(len(erode_edge)):
        x, y, w, h = cv.boundingRect(erode_edge[j])
        erode_bbox.append([x, y, x+w, y+h ,j])
    in_erode=[]
    not_in_erode=[]
    for i in range(len(initial_bbox)):
        res=iou(initial_bbox[i],erode_bbox)
        if res is None:
            not_in_erode.append(initial_bbox[i][4])
            continue
        in_erode.append(res)
    disapper_bbox=[]
    for i in range(len(not_in_erode)):
        disapper_bbox.append(initial_bbox[not_in_erode[i]])
#     bbox_index

    new_bbox=[]
    for i in range(len(erode_edge)):
        if i in in_erode:
            continue
        new_bbox.append(erode_bbox[i])
#         xmin,ymin,xmax,ymax,index
    return disapper_bbox,new_bbox


def detction_overlap_building(input_img, input_edge, kernel_size, iteration):
    img = input_img.copy()
    data = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # data = img[:, :, 0]
    _,res1, _ = cv.findContours(data, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    target_num = len(res1)
    #     kerne= np.ones((kernel_size,kernel_size),np.uint8)
    #     # left-right and top-down
    #     erosion = cv.erode(img,kernel,iterations=iteration)
    #     contours,_=cv.findContours(erosion,mode=cv.RETR_TREE,method=cv.CHAIN_APPROX_NONE)
    #     double=len(contours)
    #     暂且只考虑单方向的腐蚀操作

    img1 = input_img.copy()
    kernel = np.ones((1, kernel_size), np.uint8)
    # top-down
    erosion1 = cv.erode(img, kernel, iterations=iteration)
    # gray_e1 = erosion1[:, :, 0]
    gray_e1 = cv.cvtColor(erosion1, cv.COLOR_BGR2GRAY)
    _,contours1, _ = cv.findContours(gray_e1, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    single_td = len(contours1)

    img2 = input_img.copy()
    kernel = np.ones((kernel_size, 1), np.uint8)
    # left-right
    erosion2 = cv.erode(img2, kernel, iterations=iteration)
    #     对其进行了腐蚀之后，对于检测效果较差之处
    #gray_e2 = erosion2[:, :, 0]
    gray_e2 = cv.cvtColor(erosion2, cv.COLOR_BGR2GRAY)
    _, contours2, _ = cv.findContours(gray_e2, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    single_rl = len(contours2)

    print(target_num, single_td, single_rl, type(contours1))
    #     return data gray_e1,gray_e2,contours1

    if (single_td == target_num) and (single_rl == target_num):
        print("没有边角重叠在一起的建筑物")
        dis = None
        add = None
        k = 0
        dis1 = None
        add1 = None
        k1 = 0

    else:
        if single_td != target_num:
            print('竖直方向产生了重叠')
            print("原始的轮廓个数为：{}".format(len(res1)))
            dis, add = process_td(res1, contours1)
            bad_num = len(dis)
            if dis != []:
                for i in range(len(dis)):
                    res1[dis[i][4]] = None
            #                     不可以仅仅用pop操作，pop一个之后，全局索引已经发生了变化
            print("删除冗余之后轮廓个数为：{}".format(len(res1) - bad_num))
            if add != []:
                k = 0
                for i in range(len(add)):
                    cur_area = (add[i][2] - add[i][0]) * (add[i][3] - add[i][1])
                    if cur_area <= 50:
                        continue
                    res1.append(contours1[add[i][4]])
                    k += 1
            print("新增重叠目标之后轮廓个数为：{}".format(len(res1) - bad_num))
        else:
            dis = None
            add = None
            k = 0

        # 12/17

        if single_rl != target_num:
            print(' 水平方向产生了重叠')
            print("经过垂直方向腐蚀后轮廓的个数为:{}".format(len(res1) - bad_num))
            dis1, add1 = process_rl(res1, contours2)
            bad1_num = len(dis1)
            for i in range(bad1_num):
                res1[dis1[i][4]] = None
            print("删除水平重叠之后轮廓个数为:{}".format(len(res1)) - bad_num - bad1_num)
            k1 = 0
            for i in range(len(add1)):
                if (add1[i][2] - add1[i][0]) * (add1[i][3] - add1[i][0]) <= 50:
                    continue
                res1.append(contours2[add1[i][4]])
                k1 += 0
            print("新增水平方向新增的轮廓个数为:{}".format(len(res1) - bad_num - bad1_num))
        else:
            dis1 = None
            add1 = None
            k1 = 0
    return res1, erosion1, erosion2, dis, add, k, dis1, add1, k1


def full_fill(img, center, fill_color, lower, higher, method):
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(img, mask, center, fill_color, lower, higher, method)


for i in range(len(contours)):
    area = cv.contourArea(contours[i])
    if area == 0:
        data = contours[i].reshape((-1, 2))
        full_fill(img=initial_img, center=tuple(data[0]), fill_color=(0, 0, 0), lower=(10, 10, 10), higher=(10, 10, 10),
                  method=cv.FLOODFILL_FIXED_RANGE)
        continue
    M = cv.moments(contours[i])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if area <= 50:
        full_fill(img=initial_img, center=(cX, cY), fill_color=(0, 0, 0), lower=(10, 10, 10), higher=(10, 10, 10),
                  method=cv.FLOODFILL_FIXED_RANGE)
        continue
cv.imwrite('./full_fill.png',initial_img)
re,erode1,erode2,dis,add,k,dis1,add1,k1=detction_overlap_building(initial_img,contours,3,2)

# 图片的保存
# 图片的保存
chang_bbox=initial_img.copy()
# 竖直的overlap，disapper_bbox
if dis!=None:
    for i in range(len(dis)):
        cv.rectangle(chang_bbox,(dis[i][0],dis[i][1]),(dis[i][2],dis[i][3]),(0,255,0),10)
cv.imwrite("./change.png",chang_bbox)
cv.imwrite("./erode1.png",erode1)
cv.imwrite("./erode2.png",erode2)


# 绘制出水平方向腐蚀后多出的建筑物
for i in range(len(re)-1,len(re)-1-k1,-1):
    x,y,w,h = cv.boundingRect(re[i])
    cv.rectangle(erode2,(x,y),(x+w,y+h),(0,255,0),10)
cv.imwrite("./erode_2_bbox.png",erode2)

# 绘制竖直方向腐蚀后多出的建筑物
for i in range(len(re)-1-k1,len(re)-1-k1-k,-1):
    x,y,w,h = cv.boundingRect(re[i])
    cv.rectangle(erode1,(x,y),(x+w,y+h),(0,255,0),10)
cv.imwrite("./erode_1_bbox.png",erode1)


def small_target(edge,epsilon):
    approx = cv.approxPolyDP(edge, epsilon, True)
    points = approx.reshape((-1, 2))
    count=0
    rate=0.002
    while len(points) != 4:
        epsilon = rate * cv.arcLength(edge, True)
        rate=rate+0.002
        approx = cv.approxPolyDP(edge,epsilon,True)
        points = approx.reshape((-1, 2))
        count+=1
        if count>10:
            break
    if len(points)==4:
        print("小目标的优化结果为4边形")
        name = 'iter'
    else:
        print("小目标的优化方法为外接最小矩形")
        rect = cv.minAreaRect(edge)
        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        points = cv.boxPoints(rect)
        name = 'min'
    return points, name


def big_building(edge):
    epsilon = 0.005 * cv.arcLength(contours[i], True)
    approx = cv.approxPolyDP(contours[i],epsilon, True)
    points = approx.reshape((-1, 2))
    return points

area_result = []
contours=re
initial_img=input_img
index = 0
all_coner = []
for i in range(len(contours)):
    if contours[i] is None:
        print('*********')
        continue
    # cv.drawContours(initial_img, contours[i], -1, (0, 0, 255), 1)
    area = cv.contourArea(contours[i])
    area_result.append(int(area))
    epsilon = 0.01 * cv.arcLength(contours[i], True)

    approx = cv.approxPolyDP(contours[i], epsilon, True)
    points = approx.reshape((-1, 2))
    all_coner.append(points)
    small = False

    M = cv.moments(contours[i])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    #     提前处理了
    #     if area<=30:
    #         print("当前的建筑物为误检测点：原因是建筑物面积小于9平方米")
    #         full_fill(img=img,center=(cX,cY),fill_color=(0,0,0),lower=(1,1,1),higher=(1,1,1),method=cv.FLOODFILL_FIXED_RANGE)
    #         cv.circle(img,(cX,cY),10,(255,0,0))
    #         continue

    if area < 150:
        points, name = small_target(contours[i], epsilon=epsilon)
        small = True

    if area < 300:
        epsilon = 5 * epsilon
    if area >= 3000:
        points = big_building(contours[i])

    for j in range(len(points)):
        cv.circle(initial_img, tuple(points[j]), 1, (0, 255, 0))

    x1 = points[:, 0]
    x1 = list(x1)
    x1.append(points[0, 0])
    y1 = points[:, 1]
    y1 = list(y1)
    y1.append(points[0, 1])
    plt.plot(x1, y1)

#     if small:
#          cv.putText(img=initial_img,text=name+str(index),org=(cX,cY),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,0,255))
#     cv.putText(img=initial_img,text=str(index),org=(cX,cY),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255,0,0))
#     index+=1

# plt.imshow(initial_img, interpolation='bicubic')
plt.ylim(190,0)
# plt.savefig('00000000006.jpg', dpi=300)
plt.show()

print(all_coner)
# 所有的结果都在all_coner这个列表里面
plt_point(all_coner, name='op_pit')
