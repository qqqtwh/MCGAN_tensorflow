'''
用于存放网络
'''
import tensorflow.keras.utils
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def identity_block(x):
    dim = x.shape[-1]
    h = x

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h = Conv2D(dim, 3, padding='valid', use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = Activation('relu')(h)

    h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    h = Conv2D(dim, 3, padding='valid', use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = layers.add([x,h])
    return h

'''
生成器：先下采样再上采样
输入图片的shape=(input_height, input_width, 3）
输出的图片的shape也是(input_height, input_width, channel=3)
'''
# 官方
def get_generator2(imgshape,style_nums=5,g_Rs=9):

    dim = 64
    # 0 设置条件生成的输入格式
    img_shape = imgshape
    img_input = Input(shape=img_shape)
    label_input = Input(shape=(1,))

    label_embedding = Flatten()(Embedding(style_nums,np.prod(img_shape))(label_input))
    img_embedding = Flatten()(img_input)
    model_input = multiply([img_embedding,label_embedding])

    inputs = Reshape(img_shape)(model_input)

    # 1
    h = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = Activation('relu')(h)

    # 2
    for _ in range(2):
        dim *= 2
        h = Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization()(h)
        h = Activation('relu')(h)

    # 3
    for _ in range(g_Rs):
        h = identity_block(h)

    dim1 = dim
    dim2 = dim
    h1 = h
    h2 = h

    # 生成器1
    # 4

    for _ in range(2):
        dim1 //= 2
        h1 = Conv2DTranspose(dim1, 3, strides=2, padding='same', use_bias=False)(h1)
        h1 = InstanceNormalization()(h1)
        h1 = Activation('relu')(h1)
    # 5
    h1 = tf.pad(h1, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h1 = Conv2D(3, 7, padding='valid')(h1)
    h1 = Activation('tanh')(h1)

    # 生成器2
    # 4
    for _ in range(2):
        dim2 //= 2
        h2 = Conv2DTranspose(dim2, 3, strides=2, padding='same', use_bias=False)(h2)
        h2 = InstanceNormalization()(h2)
        h2 = Activation('relu')(h2)
    # 5
    h2 = tf.pad(h2, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h2 = Conv2D(3, 7, padding='valid')(h2)
    h2 = Activation('tanh')(h2)

    return Model([img_input,label_input],h1),Model([img_input,label_input],h2)

# 加载下采样前面
def get_generator(imgshape,style_nums=5,g_Rs=9):

    dim = 64
    # 0 设置条件生成的输入格式
    img_shape = imgshape

    img_input = Input(shape=img_shape,name='img_input')
    img_embedding = Flatten()(img_input)

    label_input = Input(shape=(1,),name='label_input')
    label_embedding = Flatten()(Embedding(style_nums,np.prod(img_shape))(label_input))

    model_input = multiply([img_embedding,label_embedding])

    inputs = Reshape(img_shape)(model_input)

    # 1
    h = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = InstanceNormalization()(h)
    h = Activation('relu')(h)

    # 双注意力
    h = cbam_block(h)

    # 2
    for _ in range(2):
        dim *= 2
        h = Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization()(h)
        h = Activation('relu')(h)

    # 3
    for _ in range(g_Rs):
        h = identity_block(h)

    dim1 = dim
    dim2 = dim
    h1 = h
    h2 = h

    # 生成器1
    # 4

    for _ in range(2):
        dim1 //= 2
        h1 = Conv2DTranspose(dim1, 3, strides=2, padding='same', use_bias=False)(h1)
        h1 = InstanceNormalization()(h1)
        h1 = Activation('relu')(h1)
    # 5
    h1 = tf.pad(h1, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h1 = Conv2D(3, 7, padding='valid')(h1)
    h1 = Activation('tanh')(h1)

    # 生成器2
    # 4
    for _ in range(2):
        dim2 //= 2
        h2 = Conv2DTranspose(dim2, 3, strides=2, padding='same', use_bias=False)(h2)
        h2 = InstanceNormalization()(h2)
        h2 = Activation('relu')(h2)
    # 5
    h2 = tf.pad(h2, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h2 = Conv2D(3, 7, padding='valid')(h2)
    h2 = Activation('tanh')(h2)

    return Model([img_input,label_input],h1),Model([img_input,label_input],h2)

# 判别器：
def get_discriminator(img_shape,style_nums=5):
    dim = 64
    # 0
    inputs =Input(shape=img_shape)

    # 1
    h = Conv2D(dim, 4, strides=2, padding='same')(inputs)
    h = LeakyReLU(alpha=0.2)(h)

    for _ in range(2):
        dim = dim * 2
        h = Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)

    dim = dim * 2

    # 判别器1
    # 2
    h1 = Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h1 = InstanceNormalization()(h1)
    h1 = LeakyReLU(alpha=0.2)(h1)
    # 3
    valid1 = Conv2D(1,4, strides=1, padding='same')(h1)
    features1 = GlobalAveragePooling2D()(h1)
    label1 = Dense(style_nums, activation='softmax')(features1)

    # 判别器2
    # 2
    h2 = Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h2 = InstanceNormalization()(h2)
    h2 = LeakyReLU(alpha=0.2)(h2)
    # 3
    valid2 = Conv2D(1, 4, strides=1, padding='same')(h2)
    features2 = GlobalAveragePooling2D()(h2)
    label2 = Dense(style_nums, activation='softmax')(features2)

    return Model(inputs=inputs, outputs=[valid1,label1]),Model(inputs=inputs, outputs=[valid2,label2])


# 通道注意力机制
def se_block(input_feature,ratio=8,name=''):
    channel = input_feature.shape[-1]

    se_feature = GlobalAvgPool2D()(input_feature)   # NHWC->NC
    se_feature = Reshape((1,1,channel))(se_feature) # NC->N11C，转为四维

    se_feature = Dense(
        channel//ratio,
        activation='relu',
        kernel_initializer='he_normal',
        use_bias=False,
        bias_initializer='zeros',
        name='se_block_'+str(name)
    )(se_feature)   # 经过relu删除了小于0的权重，代表通道的神经元个数变少

    se_feature = Dense(
        channel,
        kernel_initializer='he_normal',
        use_bias=False,
        bias_initializer='zeros',
        name='se_block_' + str(name)
    )(se_feature)   # 将大于0的权重保留后，又把代表通道的神经元个数还原

    se_feature = Activation('sigmoid')(se_feature)  # 经过sigmoid将权重规范到 0-1之间，值越大证明注意到的概率越大

    se_feature = Multiply()([input_feature,se_feature])   # 将原来的输入乘以现在的 0-1的概率值，按位逐点相乘

    return se_feature

# 通道注意力机制
def channel_attention(input_feature,ratio=8,name=''):
    channel = input_feature.shape[-1]
    avg_pool = GlobalAveragePooling2D()(input_feature)  # NHWC->NC
    max_pool = GlobalMaxPooling2D()(input_feature)  # NHWC->NC

    avg_pool = Reshape((1,1,channel))(avg_pool) # NC->N11C
    max_pool = Reshape((1,1,channel))(max_pool) # NC->N11C

    shared_layer_one = Dense(channel//ratio,activation='relu',kernel_initializer='he_normal',
                             use_bias=False,bias_initializer='zeros',name='channel_att_shared_one_'+str(name))
    shared_layer_two = Dense(channel, kernel_initializer='he_normal',
                             use_bias=False, bias_initializer='zeros', name='channel_att_shared_two_' + str(name))

    avg_pool = shared_layer_one(avg_pool)   # N11C->N11c  relu
    max_pool = shared_layer_one(max_pool)   # N11C->N11c  relu

    avg_pool = shared_layer_two(avg_pool)   # N11c->N11C  线性
    max_pool = shared_layer_two(max_pool)   # N11c->N11C  线性

    cbam_feature = Add()([avg_pool,max_pool])               # N11C 按位逐点加法
    cbam_feature = Activation('sigmoid')(cbam_feature)      # N11C  0-1

    out_feaure = Multiply()([input_feature,cbam_feature])   # N11C * NHWC->NHWC  广播机制按位逐点乘法
    return out_feaure

# 空间注意力机制
def spatial_attention(input_feature,name=''):
    ker_size = 7
    cbam_feature = input_feature
    avg_pool = Lambda(lambda x: K.mean(x,axis=3,keepdims=True))(cbam_feature)  # NHWC->NHW1
    max_pool = Lambda(lambda x: K.max(x,axis=3,keepdims=True))(cbam_feature)   # NHWC->NHW1

    concat = Concatenate()([avg_pool,max_pool]) # # NHW1-> NHW2 合成两个通道
    cbam_feature = Conv2D(1,ker_size,1,'same',
                           kernel_initializer='he_normal',use_bias=False,
                           name='spatial_att_'+str(name))(concat)               # NHW2->NHW1
    cbam_feature = Activation('sigmoid')(cbam_feature)                          # NHW2->NHW1 0-1

    out_feaure = Multiply()([input_feature,cbam_feature])   # NHW1 * NHWC->NHWC  广播机制按位逐点乘法

    return out_feaure


def cbam_block(input_feature,ratio=8,name=''):
    cbam_feature = channel_attention(input_feature,ratio,name)
    cbam_feature = spatial_attention(cbam_feature,name)
    return cbam_feature



if __name__ == '__main__':
    # g1,g2 = get_discriminator((128,128,3))
    g3,g4 = get_generator((128,128,3))
    # tensorflow.keras.utils.plot_model(g1,'./model_img/mc_att_discriminator.png')
    tensorflow.keras.utils.plot_model(g3,'./model_img/mc_att_generator.png')
    g3.summary()


