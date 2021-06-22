# -*- coding:UTF-8 -*-
"""
文件说明：
    矢量化vectorization
"""
try:
    import os
    import ogr
    import sys
    import gdal
    from findPoints import FindPoints
except Exception as e:
    print('错误原因是： ' + str(e))


def create_vector(points, img_h, shp_path):
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 支持中文路径
    gdal.SetConfigOption("SHAPE_ENCODING", "")  # 支持中文字段
    # 1、获取shape file的驱动源
    driver = ogr.GetDriverByName('ESRI Shapefile')
    #  2、创建新文件，如果这个文件存在会报错。所以可以先检查，若存在 删除
    pre_path, suf_path = os.path.split(shp_path)
    try:
        if not os.path.exists(pre_path):
            os.makedirs(pre_path)
            print('创建新的文件夹: {}', format(pre_path))
        if suf_path.split('.')[-1] not in ['shp', 'SHP']:
            print('请以".shp"为后缀名！')
            sys.exit(1)
    except Exception as reason:
        print(str(reason))
    #  './shp/hw2aa.shp'
    fn = shp_path
    if os.path.exists(fn):   # 若存在 删除
        driver.DeleteDataSource(fn)
    ds = driver.CreateDataSource(fn)   # 新建
    if ds is None:
        print('Could not create file')
        sys.exit(1)
    # 3、要添加一个新字段，只能在layer里面加，而且还不能有数据
    # 添加的字段如果是字符串，还要设定宽度
    layer = ds.CreateLayer('build', geom_type=ogr.wkbPolygon)  # 创建层
    fieldDefn = ogr.FieldDefn('ID', ogr.OFTString)  # 添加字段
    fieldDefn.SetWidth(30)  # 设置字段宽度
    layer.CreateField(fieldDefn)  # 将字段添加到shape file
    # 4、添加一个新的feature  ：  从layer中读取相应的feature类型，并创建feature
    featureDefn = layer.GetLayerDefn()
    """
    point = ogr.Geometry(ogr.wkbPoint)       创建点
    line = ogr.Geometry(ogr.wkbLineString)   创建线
    polygon = ogr.Geometry(ogr.wkbPolygon)   创建面
    """
    # 5、遍历一张图像中每一个矢量块
    # for :
    for i in range(len(points)):
        # 创建一个空的环形几何图形
        ring = ogr.Geometry(ogr.wkbLinearRing)
        # 设置这个块的名字i
        # 把顶点加到环上, 通过for循环，把一个块的点坐标加到环上
        for j in range(len(points[i][0])):
            x1 = points[i][0][j]
            y1 = points[i][1][j]
            ring.AddPoint(float(x1), float(img_h) - float(y1))
        ring.CloseRings()  # 闭合
        # 现在我们已经遍历了所有的坐标，创建一个多边形
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)  # 点坐标与创建的feature结合
        # 创建一个新特性，并设置它的几何形状和属性
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(poly)
        feature.SetField('ID', i)
        # 将该特性添加到shape file
        layer.CreateFeature(feature)
        # 破坏几何形状和特征
        ring.Destroy()
        poly.Destroy()
        feature.Destroy()
    ds.Destroy()
