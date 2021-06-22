try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import sys
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import *
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import *
    from tensorflow.keras import backend as keras
    from tensorflow.keras.preprocessing.image import img_to_array
    import random
    import cv2 as cv
except ModuleNotFoundError as reason:
    print('错误原因是：' + str(reason))


def load_img(path, grayscale=False):
    if grayscale:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = np.array(img, dtype='float') / 255
    else:
        img = cv.imread(path)
        img = np.array(img, dtype='float') / 127.5 - 1
    return img


def get_train_val(imgpath, val_rate=0.1):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(imgpath):
        train_url.append(pic)
    total_num = len(train_url)
    val_num = int(total_num * val_rate)
    random.shuffle(train_url)

    for i in range(total_num):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


def generateData(imgpath, labpath, batch_size, data):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in range(len(data)):
            url = data[i]
            batch += 1
            img = load_img(imgpath + '/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(labpath + '/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


def generateValidData(imgpath, labpath, batch_size, data):
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(imgpath + '/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(labpath + '/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


def train_model(m_Inputimgspath, m_Inputlabspath, m_Outmodelpath, epo, bs, initepo, inputmodel):
    print('/************ Again train model *******************/')
    sys.stdout.flush()
    EPOCHS = int(epo)
    BS = int(bs)
    inepo = int(initepo)
    img_path = m_Inputimgspath
    lab_path = m_Inputlabspath
    model_path = m_Outmodelpath
    train_set, val_set = get_train_val(imgpath=img_path, val_rate=0.1)
    train_num = int(len(train_set) / BS)
    val_num = int(len(val_set) / BS)
    myGene = generateData(imgpath=img_path,labpath=lab_path, batch_size=BS, data=train_set)
    myGene_test = generateValidData(imgpath=img_path,labpath=lab_path, batch_size=BS, data=val_set)
    model = load_model(inputmodel)
    # history = model.fit_generator(myGene, steps_per_epoch=train_num, epochs=EPOCHS, verbose=1,
    #                               validation_data=myGene_test, validation_steps=val_num)
    model.fit_generator(myGene,
                        steps_per_epoch=train_num,
                        epochs=EPOCHS + inepo,
                        verbose=1,
                        initial_epoch=inepo,
                        validation_data=myGene_test,
                        validation_steps=val_num)
    print('/**********   Save the model   ************/')
    sys.stdout.flush()
    model.save(model_path)


if __name__ == '__main__':
    # train_model(m_Inputimgspath=r'G:\data\img',
    #             m_Inputlabspath=r'G:\data\lab',
    #             m_Outmodelpath=r'G:\data\1.h5',
    #             epo=1,
    #             bs=1,
    #             initepo=1,
    #             inputmodel=r'D:\PIEDemo\python37\1.h5')
    try:
        train_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    except Exception as e:
        print('错误原因是： ' + str(e))