import cv2
from glob import glob
import numpy as np


class DataLoader():
    # style_names 为风格的名称，第0个是真实照片，标签为label0
    # 其余的第i个为风格图片，标签为labeli
    def __init__(self,style_names,img_res=(128, 128)):
        self.style_names = style_names
        self.img_res = img_res

    # 返回3通道的RGB格式图片
    def imread(self, path):
        img = cv2.imread(path)  # opencv读取图片时默认返回3通道BGR格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # 模型训练时加载原始照片和风格图片
    def load_batch(self,batch_size=1):
        # 将所有风格的标签和对应的图片路径保存到字典中
        all_img_paths = {}
        for label,style_name in enumerate(self.style_names):
            all_img_paths.setdefault(label,glob('./datasets/'+style_name+'/*'))
        # 获得多种风格图片的最小图片数
        picture_nums = []
        for key in all_img_paths.keys():
            picture_num = len(all_img_paths.get(key))
            picture_nums.append(picture_num)

        self.n_batches = int(min(picture_nums) / batch_size)  # 确保最后是整数个样本
        total_samples = self.n_batches * batch_size

        # 从每个风格中无放回随机选取 total_samples 个图片
        total_samples_img_paths = {}
        for key in all_img_paths.keys():
            paths = np.random.choice(all_img_paths.get(key), size=total_samples, replace=False)
            total_samples_img_paths.setdefault(key,paths)

        # 每次选取batch_size个图片送到模型中训练
        for i in range(self.n_batches):
            label_patch_imgs = {}
            # 遍历字典，将每个风格的batch_size个图片处理后放入新字典
            for label in total_samples_img_paths.keys():
                img_paths = total_samples_img_paths.get(label)[i * batch_size:(i + 1) * batch_size]
                imgs = []
                for img_path in img_paths:
                    img = self.imread(img_path)
                    img = cv2.resize(img, self.img_res)
                    if np.random.random() > 0.5:
                        img = np.fliplr(img)
                    imgs.append(img)
                imgs = np.array(imgs, dtype=np.float32) / 127.5 - 1
                label_patch_imgs.setdefault(label,imgs)
            yield label_patch_imgs

    # 随机选取batch_size个所有风格的图片
    def load_img(self,batch_size=1):
        # 将所有风格的标签和对应的图片路径保存到字典中
        all_img_paths = {}
        for label,style_name in enumerate(self.style_names):
            all_img_paths.setdefault(label,glob('./datasets/'+style_name+'/*'))
        # 每次选取batch_size个图片送到模型中训练
        label_patch_imgs = {}
        for i in range(0,len(self.style_names)):
            img_paths = np.random.choice(all_img_paths.get(i), size=batch_size, replace=True)
            imgs = []
            for img_path in img_paths:
                img = self.imread(img_path)
                img = cv2.resize(img, self.img_res)
                if np.random.random() > 0.5:
                    img = np.fliplr(img)
                imgs.append(img)
            imgs = np.array(imgs, dtype=np.float32) / 127.5 - 1
            label_patch_imgs.setdefault(i, imgs)
        return label_patch_imgs



if __name__ == '__main__':
    pass




