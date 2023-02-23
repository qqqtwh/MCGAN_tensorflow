import numpy as np
import tensorflow.keras.models

from get_resnet import get_generator
import cv2
import os
import matplotlib.pyplot as plt

# 返回3通道的RGB格式图片,并且为[N,H,W,C]格式
def imread(path,img_size):
    img = cv2.imread(path)  # opencv读取图片时默认返回3通道BGR格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size,img_size))
    img = np.array(img, dtype=np.float32) / 127.5 - 1
    # [-1,1]
    return img
def show_img(img,img_size):
    print(img)

    a = np.ones(img.shape)
    img = np.array((a+img)*127.5,dtype=np.int)
    img = cv2.resize(img, (img_size, img_size))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_img2(img,img_size):
    img = 0.5*img+0.5
    img = cv2.resize(img, (img_size, img_size))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def save_img(target_img, title, dsize):
    my_dpi = 300

    target_img = np.squeeze(target_img,axis=0)
    target_imgn = cv2.resize(target_img,(dsize,dsize))


    plt.figure(figsize=((dsize)/my_dpi,(dsize)/my_dpi))
    plt.subplots_adjust(0,0,1,1,0,0)
    plt.title(title)
    plt.axis('off')
    os.makedirs('testimg',exist_ok=True)
    plt.imshow(target_imgn)
    plt.savefig('testimg/'+title+'.png',dpi=my_dpi)
    plt.close()

if __name__ == '__main__':
    # 1.参数初始化
    # sn中的风格名称要和训练时的顺序保持一致
    sn = [
        # ['photo', 'ukiyoe', 'vangogh'],
        ['photo', 'cezanne','monet', 'ukiyoe', 'vangogh']
    ]
    img_rc = 128  # 图像的宽高
    g_Rs = 6  # 生成器残差快的数量(128及以下适用6，256及以上适用9)
    last_epoch = 376     # 模型最合适的训练序号
    # 2.获取模型参数路径
    style_names = sn[0]
    weights_name = ''
    for i in range(1, len(style_names)):
        weights_name += style_names[i] + '_'
    weights_name += str(img_rc) + '_' + str(g_Rs)
    if weights_name not in os.listdir('download_weights'):
        print('model does not exist,please modify the initial parameters')
    else:
    # 3.获取原图
        img0 = imread('testimg/111.jpg',img_rc)
        img0 = np.expand_dims(img0,0)
    # 4.创建模型并加载参数
        g_IO,g_OI = get_generator((img_rc,img_rc,3),len(style_names),g_Rs)

        g_IO.load_weights('download_weights/%s/g_IO_epoch_%d/' % (weights_name, int(last_epoch)))
        g_OI.load_weights('download_weights/%s/g_OI_epoch_%d/' % (weights_name, int(last_epoch)))

    # 5.预测
        dsize = 128
        save_img(0.5*img0+0.5, 'photo_'+str(last_epoch), dsize)
        for i in range(1,len(style_names)):
            label = np.array([i])
            target_img = g_IO.predict([img0,label])
            target_img = 0.5*target_img+0.5
            title = style_names[i]+'_' + str(last_epoch)
            save_img(target_img,title,dsize)






