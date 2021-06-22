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


def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation('relu')(x)
    return x


def conv_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tf.keras.layers.Add()([conv, shortcut])
    return output


def residual_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    conv = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tf.keras.layers.Add()([shortcut, conv])
    return output


def up_sampling_concat_block(x, x_skip):
    up_sample = tf.keras.layers.UpSampling2D((2, 2))(x)
    concat = tf.keras.layers.Concatenate()([up_sample, x_skip])
    return concat


def ResUnet(imgw, imgh, cla_num):
    f = [16, 32, 64, 128, 256]
    inputs = tf.keras.layers.Input((imgw, imgh, 3))

    # encoding
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    # bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    # decoder
    u0 = up_sampling_concat_block(b1, e4)
    d0 = residual_block(u0, f[4])

    u1 = up_sampling_concat_block(d0, e3)
    d1 = residual_block(u1, f[3])

    u2 = up_sampling_concat_block(d1, e2)
    d2 = residual_block(u2, f[2])

    u3 = up_sampling_concat_block(d2, e1)
    d3 = residual_block(u3, f[1])

    output = tf.keras.layers.Conv2D(cla_num, (1, 1), padding='same', activation='softmax')(d3)
    model = tf.keras.models.Model(inputs, output)
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def load_img(path, grayscale=False):
    if grayscale:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = np.array(img, dtype='float')/255
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


def train_model(m_Inputimgspath, m_Inputlabspath, m_Outmodelpath, m_w, m_h, epo, bs, class_num):
    print('/************ train resunet model *******************/')
    sys.stdout.flush()
    EPOCHS = int(epo)
    BS = int(bs)
    img_path = m_Inputimgspath
    lab_path = m_Inputlabspath
    model_path = m_Outmodelpath
    img_w = int(m_w)
    img_h = int(m_h)
    train_set, val_set = get_train_val(imgpath=img_path, val_rate=0.1)
    train_num = int(len(train_set) / BS)
    val_num = int(len(val_set) / BS)
    myGene = generateData(imgpath=img_path,labpath=lab_path, batch_size=BS, data=train_set)
    myGene_test = generateValidData(imgpath=img_path,labpath=lab_path, batch_size=BS, data=val_set)
    model = ResUnet(imgw=img_w, imgh=img_h,cla_num=class_num)
    history = model.fit_generator(myGene, steps_per_epoch=train_num, epochs=EPOCHS, verbose=1,
                                  validation_data=myGene_test, validation_steps=val_num)
    print('/**********   Save the model   ************/')
    sys.stdout.flush()
    model.save(model_path)


if __name__ == '__main__':
    try:
        train_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    except Exception as e:
        print('错误原因是： ' + str(e))