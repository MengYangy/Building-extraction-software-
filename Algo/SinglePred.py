try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import cv2 as cv
    import random, glob
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import sys
except ModuleNotFoundError as reason:
    print('错误原因是：' + str(reason))


# def img_pred(imgpath, modelpath, savepath, img_h_w):
#     print('/************ 加载模型 *******************/')
#     model = load_model(modelpath)
#     stride = int(img_h_w)
#     image_size = int(img_h_w)
#     print('/************ 图片预测 *******************/')
#     for img in imgpath.split(';')[:-1]:
#         name = img.split('/')[-1].split('.')[0]
#
#         #     image=tf.io.read_file(images[im])
#         #     image=tf.image.decode_png(image,channels=3)
#         image = cv.imread(img)
#         image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#         h, w, c = image.shape
#         if h % stride != 0:
#             padding_h = (h // stride + 1) * stride
#         else:
#             padding_h = (h // stride) * stride
#         if w % stride != 0:
#             padding_w = (w // stride + 1) * stride
#         else:
#             padding_w = (w // stride) * stride
#         padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
#         padding_img[0:h, 0:w, :] = image[:, :, :]
#         mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
#
#         for i in range(padding_h // stride):
#             print(i)
#             for j in range(padding_w // stride):
#                 crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :3]
#                 pred_result = np.ones((image_size, image_size), np.int8)
#                 crop = tf.cast(crop, tf.float32)
#                 test_part = crop
#                 test_part = test_part / 127.5 - 1
#                 test_part = tf.expand_dims(test_part, axis=0)
#                 pred_part = model.predict(test_part)
#                 pred_part = tf.argmax(pred_part, axis=-1)
#                 pred_part = pred_part[..., tf.newaxis]
#                 pred_part = tf.squeeze(pred_part)
#                 pred_result = pred_part.numpy()
#                 mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred_result[:h, :w]
#         cv.imwrite(os.path.join(savepath, name + '_pred.tif'), mask_whole)
#         print('完成图像预测')



class Pred_Module():
    def __init__(self):
        print('/************ 加载模型 *******************/')

    def imgs_pred(self, img_paths, save_path):
        print('/************ 图片预测 *******************/')
        for img_path in img_paths.split(';')[:-1]:
            name = img_path.split('/')[-1].split('.')[0]
            img_name = os.path.join(save_path, name + '_pred.tif')
            self.pred_func(img_path, img_name)
            print('{}预测完成 -> 结果保存为{}'.format(img_path, img_name))

    def dir_pred(self, img_paths, save_path):
        for img_path in glob.glob(img_paths + '/*'):
            name = img_path.split('/')[-1].split('.')[0]
            img_name = os.path.join(save_path, name + '_pred.tif')
            self.pred_func(img_path, img_name)
            print('{}预测完成 -> 结果保存为{}'.format(img_path, img_name))

    def load_model(self, model_path, img_h_w):
        self.model = load_model(model_path)
        self.stride = int(img_h_w)
        self.image_size = int(img_h_w)

    def pred_func(self, img, img_name):
        image_size = stride = self.stride
        image = cv.imread(img)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        h, w, c = image.shape
        if h % stride != 0:
            padding_h = (h // stride + 1) * stride
        else:
            padding_h = (h // stride) * stride
        if w % stride != 0:
            padding_w = (w // stride + 1) * stride
        else:
            padding_w = (w // stride) * stride
        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)

        for i in range(padding_h // stride):
            print(i)
            for j in range(padding_w // stride):
                crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :3]
                pred_result = np.ones((image_size, image_size), np.int8)
                crop = tf.cast(crop, tf.float32)
                test_part = crop
                test_part = test_part / 127.5 - 1
                test_part = tf.expand_dims(test_part, axis=0)
                pred_part = self.model.predict(test_part)
                pred_part = tf.argmax(pred_part, axis=-1)
                pred_part = pred_part[..., tf.newaxis]
                pred_part = tf.squeeze(pred_part)
                pred_result = pred_part.numpy()
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred_result[:h, :w]
        cv.imwrite(img_name, mask_whole)




#
# if __name__ == '__main__':
#     # img_pred(imgpath=r'C:\Users\Administrator\Desktop\WeData\test\1.png',
#     # savepath=r'C:\Users\Administrator\Desktop\WeData\pred\pred_1.png',
#     #       modelpath=r'C:\Users\Administrator\Desktop\WeData\model\wedata_unet_v1.h5')
#     try:
#         img_pred(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
#     except Exception as e:
#         print('错误原因是： ' + str(e))
