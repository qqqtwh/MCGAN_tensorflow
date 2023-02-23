import tensorflow.keras.models

from McDataLoader import DataLoader
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from get_resnet import get_generator, get_discriminator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class MCGAN():

    def build_adv(self, lambda_clcye, lambda_id, style_nums):
        # 根据风格种类数(包含真实照片)创建图片输入和标签输入
        imgs = []
        labels = []
        my_inputs = []
        for i in range(style_nums):
            imgs.append(Input(shape=self.img_shape))
            labels.append(Input(shape=(1,)))
        for elem in imgs:
            my_inputs.append(elem)
        for elem in labels:
            my_inputs.append(elem)

        # 根据CycleGan的损失组成，生成所需的数据
        valids = []  # 普通损失的patch判断结果
        target_labels = []  # 普通损失的标签判断结果
        reconstrs = []  # 循环一直性生成的图片
        img_i_ids = []  # Identity loss 生成的图片
        fake_styles = [] # 所有生成的假风格图片
        for i in range(1, style_nums):
            # I-g_IO>O'
            fake_i = self.g_IO([imgs[0], labels[i]])
            # O-g_OI>I'
            fake_Ii = self.g_OI([imgs[i], labels[0]])
            # O-g_OI>I'-g_IO>O'
            reconstr_i = self.g_IO([fake_Ii, labels[i]])
            # I-g_IO>O'-g_OI->I'
            reconstr_Ii = self.g_OI([fake_i, labels[0]])
            # O-g_IO>O'
            img_i_id = self.g_IO([imgs[i], labels[i]])
            # 判断真假
            valid_i, target_labeli = self.d_O(fake_i)
            valid_Ii, target_labelIi = self.d_I(fake_Ii)

            valids.append(valid_i)  # 假风格图片的真假判断结果
            valids.append(valid_Ii)  # 假原图的真假判断结果
            target_labels.append(target_labeli)  # 假风格图片的标签判断结果
            target_labels.append(target_labelIi)  # 假原图的标签判断结果
            reconstrs.append(reconstr_i)  # 循环一致生成的假风格图片
            reconstrs.append(reconstr_Ii)  # 循环一致生成的假原图图片
            img_i_ids.append(img_i_id)  # 根据 Identity loss 生成的风格图片

        # I-g_OI>I' 根据 Identity loss 生成的原图
        img_0_id = self.g_OI([imgs[0], labels[0]])
        # 封装模型输出 和 模型编译的损失和损失权重

        '''
        valids内容:[valid_1,valid_01,valid_2,valid_02,...]
        target_labels内容:[target_label1,target_label01,target_label2,target_label02,...]
        reconstrs内容:[reconstr_1,reconstr_01,reconstr_2,reconstr_02,...]
        img_i_ids内容:[img_1_id,img_2_id,...]
        最后 all_outputs的内容顺序应该是 [
            valid1,valid_01,valid2,valid_02,...
            target_label1,target_label01,target_label2,target_label02,...
            reconstr_1,reconstr_01,reconstr_2,reconstr_02,...
            img_1_id,img_2_id,...
            img_0_id]
        '''
        all_losses = []
        all_loss_weights = []
        my_outputs = []
        # 真假对抗损失
        for elem in valids:
            all_losses.append('mse')
            all_loss_weights.append(0.5)
            my_outputs.append(elem)
        # 标签损失
        for elem in target_labels:
            all_losses.append('sparse_categorical_crossentropy')
            all_loss_weights.append(0.5)
            my_outputs.append(elem)

        # 循环损失
        for elem in reconstrs:
            all_losses.append('mae')
            all_loss_weights.append(lambda_clcye)
            my_outputs.append(elem)

        # 一致性损失
        for elem in img_i_ids:
            all_losses.append('mae')
            all_loss_weights.append(lambda_id)
            my_outputs.append(elem)
        my_outputs.append(img_0_id)
        all_losses.append('mae')
        all_loss_weights.append(lambda_id)


        # 对抗网络
        combined = Model(inputs=my_inputs, outputs=my_outputs)
        combined.compile(loss=all_losses, optimizer=self.optimizer, loss_weights=all_loss_weights)
        return combined

    def __init__(self, style_names, img_rc=256, g_Rs=9,batch_size=1):

        # 一.设置图片shape
        self.img_rows = img_rc
        self.img_cols = img_rc
        self.channels = 3
        self.g_Rs = g_Rs
        self.style_names = style_names
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dir_name = ''
        for i in range(1, len(style_names)):
            self.dir_name += style_names[i] + '_'
        self.dir_name += str(img_rc) + '_' + str(g_Rs)+'_'+str(batch_size)

        # 二.数据加载器
        self.data_loader = DataLoader(style_names, img_res=(self.img_rows, self.img_cols))

        # 三.判别器形状
        patch = int(self.img_rows / (2 ** 3))
        self.disc_path = (patch, patch, 1)
        # 四.损失风格权重
        self.lambda_cycle = 10
        self.lambda_id = 0.5
        # 五.优化器
        self.optimizer = Adam(0.0002, 0.5)  # 和cyclegan保持一致
        self.style_nums = len(style_names)

        # 六.判别器的创建编译
        # 2个判别器
        losses = ['mse', 'sparse_categorical_crossentropy']
        self.d_I, self.d_O = get_discriminator(self.img_shape, self.style_nums)
        self.d_I.compile(loss=losses, optimizer=self.optimizer, metrics=['accuracy'])
        self.d_O.compile(loss=losses, optimizer=self.optimizer, metrics=['accuracy'])

        # 2个生成器
        self.g_IO, self.g_OI = get_generator(self.img_shape, self.style_nums, self.g_Rs)
        self.d_I.trainable = False
        self.d_O.trainable = False
        # 七.对抗生成网络的创建编译
        self.combined = self.build_adv(self.lambda_cycle, self.lambda_id, self.style_nums)

    def scheduler(self, models):
        # 每隔100epoch将学习率缩小一半
        temp = K.get_value(models[0].optimizer.lr)
        for model in models:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.5)
        return temp * 0.5

    def train(self, init_epoch, epochs, batch_size=1, sample_interval=50):

        # 生成 batch_size 个 path格式的评委
        valid = np.ones((batch_size,) + self.disc_path)
        fake = np.zeros((batch_size,) + self.disc_path)

        if init_epoch != 0:
            self.d_I.load_weights('mymodels/%s/d_I_epoch_%d/' % (self.dir_name, int(init_epoch)))
            self.d_O.load_weights('mymodels/%s/d_O_epoch_%d/' % (self.dir_name, int(init_epoch)))
            self.g_IO.load_weights('mymodels/%s/g_IO_epoch_%d/' % (self.dir_name, int(init_epoch)))
            self.g_OI.load_weights('mymodels/%s/g_OI_epoch_%d/' % (self.dir_name, int(init_epoch)))

        for epoch in range(init_epoch, epochs):
            # 调整学习率
            if epoch % 100 == 0 and epoch != 0:
                os.makedirs('./myloggers/' + self.dir_name, exist_ok=True)
                lr_messages = open('./myloggers/' + self.dir_name + '/lr_messages.txt', 'a')
                now_lr = self.scheduler([self.combined, self.d_I, self.d_O])
                print('lr change to:', now_lr)
                print('lr change to:', now_lr, file=lr_messages)
                lr_messages.close()
            start_time = datetime.datetime.now()
            # 根据batch_size每一个epoch分成多个batch_i,每个batch_i训练时，每种风格的图片各取batch_size个
            for batch_i, imgsdict in enumerate(self.data_loader.load_batch(batch_size)):

                # 1.创建真实的标签和图片，每种风格对应 batch_size 个图片和标签
                labels = []
                imgs = []
                for i in range(len(imgsdict)):
                    if i == 0:
                        labels.append(np.array([[0]]*batch_size))
                    else:
                        mos = np.array([[1*i]]*batch_size)
                        labels.append(mos)
                    imgs.append(imgsdict.get(i))
                my_inputs = []
                for elem in imgs:
                    my_inputs.append(elem)
                for elem in labels:
                    my_inputs.append(elem)

                # 2.训练生成模型
                # g_loss = [总loss,n个输出的每个loss]
                # n = (style_nums-1)*2*3 + style_nums
                # n 分别对应 validi,valid0i,...t_labeli,t_label0i,...,resi,res0i,...,idi,...id0

                my_outputs = []
                # 加入所有的patch判断结果，共(styles-1)*2个
                for i in range(1,self.style_nums):
                    my_outputs.append(valid)
                    my_outputs.append(valid)

                # 加入所有的 label判断结果[1,0,2,0,3,0...]
                for i in range(1,self.style_nums):
                    my_outputs.append(labels[i])
                    my_outputs.append(labels[0])

                # 加入所有的重构图片结果
                for i in range(1, self.style_nums):
                    my_outputs.append(imgs[i])
                    my_outputs.append(imgs[0])
                # 加入所有的反转图片结果
                for i in range(1, self.style_nums):
                    my_outputs.append(imgs[i])
                my_outputs.append(imgs[0])

                g_loss = self.combined.train_on_batch(my_inputs,my_outputs)

                fake_imgs = [[]]
                fakeI_imgs = [[]]
                for i in range(1,self.style_nums):
                    # 先生成假图片
                    fake_i = self.g_IO.predict([imgs[0],labels[i]])
                    fake_Ii = self.g_OI.predict([imgs[i],labels[0]])
                    fake_imgs.append(fake_i)
                    fakeI_imgs.append(fake_Ii)


                # 4.训练判别器 d_loss = [loss,val_loss,tl_loss,valid_accuracy,label_accuracy]
                d_O_loss_reals = []
                d_O_loss_fakes = []
                # 训练判别器 d_O
                for i in range(1,self.style_nums):
                    d_O_loss_real = self.d_O.train_on_batch(imgs[i],[valid,labels[i]])
                    d_O_loss_reals.append(d_O_loss_real)
                    d_O_loss_fake = self.d_O.train_on_batch(fake_imgs[i],[fake,labels[i]])
                    d_O_loss_fakes.append(d_O_loss_fake)
                d_O_loss = 0.5*np.add(np.mean(d_O_loss_reals,axis=0),np.mean(d_O_loss_fakes,axis=0))

                d_I_loss_reals = []
                d_I_loss_fakes = []
                # 训练判别器 d_I
                for i in range(1,self.style_nums):
                    d_I_loss_fake = self.d_I.train_on_batch(fakeI_imgs[i],[fake,labels[0]])
                    d_I_loss_fakes.append(d_I_loss_fake)
                d_I_loss_real = self.d_I.train_on_batch(imgs[0],[valid,labels[0]])
                d_I_loss_reals.append(d_I_loss_real)
                d_I_loss = 0.5 * np.add(np.mean(d_I_loss_reals, axis=0), np.mean(d_I_loss_fakes, axis=0))

                d_loss = 0.5 * np.add(d_O_loss, d_I_loss)

                batch_i_time = datetime.datetime.now() - start_time
                
                os.makedirs('./myloggers/'+self.dir_name, exist_ok=True)

                temp_index = (self.style_nums - 1)*2
                print('[Epoch %d/%d] [Batch %d/%d]'%(epoch,epochs,batch_i,self.data_loader.n_batches))
                print('[D_loss: %f, acc: %3d,val_acc: %3d, tl_acc: %3d]' %(
                    d_loss[0],np.mean([100 * d_loss[3],100 * d_loss[4]]),100*d_loss[3],100*d_loss[4]))
                print('[G_loss: %05f, val_loss: %05f, tl_loss: %05f, rec_loss: %05f, id_loss: %05f] time: %s' %
                    (g_loss[0],
                      np.mean(g_loss[1:temp_index+1]),
                      np.mean(g_loss[temp_index+1:temp_index*2+1]),
                      np.mean(g_loss[temp_index*2+1:temp_index*3+1]),
                      np.mean(g_loss[temp_index*3+1:-1]),
                      batch_i_time))
                # 记录训练信息
                self.log_message(epoch, epochs, batch_i, d_loss, g_loss, batch_i_time, temp_index)


                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
            if epoch%2==0 and epoch!=init_epoch:
                os.makedirs('mymodels/'+self.dir_name, exist_ok=True)
                self.d_I.save_weights('mymodels/%s/d_I_epoch_%d/' %(self.dir_name,int(epoch)))
                self.d_O.save_weights('mymodels/%s/d_O_epoch_%d/' %(self.dir_name,int(epoch)))
                self.g_IO.save_weights('mymodels/%s/g_IO_epoch_%d/' %(self.dir_name,int(epoch)))
                self.g_OI.save_weights('mymodels/%s/g_OI_epoch_%d/' %(self.dir_name,int(epoch)))

    def log_message(self,epoch,epochs,batch_i,d_loss,g_loss,batch_i_time,temp_index):
        train_d_loss = open('./myloggers/' + self.dir_name + '/train_d_loss.txt', 'a')
        train_acc = open('./myloggers/' + self.dir_name + '/train_acc.txt', 'a')
        train_val_acc = open('./myloggers/' + self.dir_name + '/train_val_acc.txt', 'a')
        train_tl_acc = open('./myloggers/' + self.dir_name + '/train_tl_acc.txt', 'a')
        train_g_loss = open('./myloggers/' + self.dir_name + '/train_g_loss.txt', 'a')
        train_val_loss_tl_loss = open('./myloggers/' + self.dir_name + '/train_val_loss_tl_loss.txt', 'a')
        train_rec_loss_id_loss = open('./myloggers/' + self.dir_name + '/train_rec_loss_id_loss.txt', 'a')

        print('[Epoch %d/%d] [Batch %d/%d] [D_loss: %f]  time: %s' % (
            epoch, epochs, batch_i, self.data_loader.n_batches, d_loss[0], batch_i_time), file=train_d_loss)
        print('[Epoch %d/%d] [Batch %d/%d] [acc: %3d]  time: %s' % (
            epoch, epochs, batch_i, self.data_loader.n_batches, np.mean([100 * d_loss[3], 100 * d_loss[4]]),
            batch_i_time), file=train_acc)
        print('[Epoch %d/%d] [Batch %d/%d] [val_acc: %3d]  time: %s' % (
            epoch, epochs, batch_i, self.data_loader.n_batches, 100 * d_loss[3], batch_i_time), file=train_val_acc)
        print('[Epoch %d/%d] [Batch %d/%d] [tl_acc: %3d]  time: %s' % (
            epoch, epochs, batch_i, self.data_loader.n_batches, 100 * d_loss[4], batch_i_time), file=train_tl_acc)

        print('[Epoch %d/%d] [Batch %d/%d] [G_loss: %05f]  time: %s' % (
            epoch, epochs, batch_i, self.data_loader.n_batches, g_loss[0], batch_i_time), file=train_g_loss)
        print('[Epoch %d/%d] [Batch %d/%d] [val_loss: %05f, tl_loss: %05f]  time: %s' % (
            epoch, epochs, batch_i, self.data_loader.n_batches, np.mean(g_loss[1:temp_index + 1]),
            np.mean(g_loss[temp_index + 1:temp_index * 2 + 1]), batch_i_time), file=train_val_loss_tl_loss)
        print('[Epoch %d/%d] [Batch %d/%d] [rec_loss: %05f, id_loss: %05f]  time: %s' % (
            epoch, epochs, batch_i, self.data_loader.n_batches,
            np.mean(g_loss[temp_index * 2 + 1:temp_index * 3 + 1]), np.mean(g_loss[temp_index * 3 + 1:-1]),
            batch_i_time), file=train_rec_loss_id_loss)

        train_d_loss.close()
        train_acc.close()
        train_val_acc.close()
        train_tl_acc.close()
        train_g_loss.close()
        train_val_loss_tl_loss.close()
        train_rec_loss_id_loss.close()


    def mydraw(self, axs, i, j, name, counts, gen_imgs, fontsize=7):
        axs[i, j].imshow(gen_imgs[counts][0])
        axs[i, j].set_title(name, fontsize=fontsize)
        axs[i, j].axis('off')
        return 1  # 方便counts加1

    def get_img_titles(self):
        # 第一行图片名称
        img_titles = ['photo']
        n = len(self.style_names)
        # 第二行图片名称
        for i in range(1,n):
            img_titles.append('photo->' + self.style_names[i])
        # 第三行图片名称
        for i in range(1,n):
            img_titles.append('photo->' + self.style_names[i]+'->photo')
        # 第四行图片名称
        for i in range(1,n):
            img_titles.append(self.style_names[i])
        # 第五行图片名称
        for i in range(1,n):
            img_titles.append(self.style_names[i]+'->photo')
        # 第六行图片名称
        for i in range(1,n):
            img_titles.append(self.style_names[i]+'->photo->'+ self.style_names[i])

        return img_titles

    def draw_sing_img(self,genimgs,img_titles,epoch, batch_i):
        path = 'singImages/' + self.dir_name + '/' + str(epoch) + '_' + str(batch_i)
        os.makedirs(path, exist_ok=True)
        counts = 0
        my_dpi = 300

        for genimg in genimgs:
            plt.figure(figsize=((self.img_rows) / my_dpi, (self.img_rows) / my_dpi))
            plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            title = img_titles[counts]
            plt.title(title, fontsize=17)
            plt.axis('off')
            plt.imshow(genimg[0])
            plt.savefig(path + '/' + str(counts) + '.png',dpi=my_dpi)
            plt.close()
            counts+=1

    def get_gen_imgs(self):
        # 1.读取图片
        batch_size = 1
        imgsdict = self.data_loader.load_img(batch_size)
        labels = []
        imgs = []
        for i in range(len(imgsdict)):
            if i == 0:
                labels.append(np.array([0] * batch_size))
            else:
                mos = np.array([1 * i] * batch_size)
                labels.append(mos)
            imgs.append(imgsdict.get(i))
        # 2.将图片传入模型
        fakess, fakes2ps, fakeps, fakep2ss = [], [], [], []
        for i in range(1, self.style_nums):
            # p-s'
            fakes = self.g_IO([imgs[0], labels[i]])
            fakess.append(np.array(fakes))
            # p-s'-p'
            fakes2p = self.g_OI([fakes, labels[0]])
            fakes2ps.append(np.array(fakes2p))
            # s-p'
            fakep = self.g_OI([imgs[i], labels[0]])
            fakeps.append(np.array(fakep))
            # s-p'-s''
            fakep2s = self.g_IO([fakep, labels[i]])
            fakep2ss.append(np.array(fakep2s))

        gen_imgs = []

        gen_imgs.append(imgs[0])

        for elem in fakess:
            gen_imgs.append(elem)
        for elem in fakes2ps:
            gen_imgs.append(elem)

        for i in range(1, self.style_nums):
            gen_imgs.append(imgs[i])
        for elem in fakeps:
            gen_imgs.append(elem)
        for elem in fakep2ss:
            gen_imgs.append(elem)

        gen_imgs = np.array(gen_imgs, dtype=np.float)
        gen_imgs = 0.5 * gen_imgs + 0.5
        return gen_imgs

    def draw_all_imgs(self,gen_imgs,img_titles,epoch, batch_i):
        path = 'images/' + self.dir_name
        os.makedirs(path, exist_ok=True)
        r, c = 6, self.style_nums - 1
        counts = 0
        my_dpi = 300
        plt.figure(figsize=((self.img_rows) / my_dpi, (self.img_rows) / my_dpi))
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        fig, axs = plt.subplots(r, c, constrained_layout=True)

        if c == 1:
            axs = axs[:, np.newaxis]

        for i in range(r):
            if i == 0:
                counts += self.mydraw(axs, i, 0, img_titles[counts], counts, gen_imgs)
                for cc in range(1, c):
                    axs[i, cc].axis('off')
            else:
                for j in range(c):
                    counts += self.mydraw(axs, i, j, img_titles[counts], counts, gen_imgs)
        fig.savefig(path + '/%d_%d.png' % (epoch, batch_i),dpi=my_dpi)
        plt.close()

    def sample_images(self, epoch, batch_i):
        # 1.获取图片
        gen_imgs = self.get_gen_imgs()
        img_titles = self.get_img_titles()
        # 2.画在一张图上
        self.draw_all_imgs(gen_imgs,img_titles,epoch,batch_i)

        # 3.画单个图
        self.draw_sing_img(gen_imgs,img_titles,epoch,batch_i)



if __name__ == '__main__':
    img_rc = 128  # 图像的宽高
    g_Rs = 6  # 生成器残差快的数量(128及以下适用6，256及以上适用9)
    batch_size = 2
    style_names = ['photo', 'cezanne', 'monet','ukiyoe','vangogh']
    mcgan = MCGAN(style_names, img_rc, g_Rs,batch_size)
    mcgan.train(init_epoch=0, epochs=1000, batch_size=batch_size)
