from torch.utils.data import Dataset
import glob
import numpy as np
import cv2
import h5py


# https://blog.csdn.net/weixin_49703603/article/details/112259517

class Dataset(Dataset):
    def __init__(self, flip_prob=None, crop_type=None, crop_size=0):

        # 参数初始化,是否需要随机翻转,是否需要随机裁减
        self.flip_prob = flip_prob
        self.crop_type = crop_type
        self.crop_size = crop_size

        # 需要读取的文件路径及文件名称
        data_path = './data/'  # 记得更改
        data_file = 'nyu_depth_v2_labeled.mat'

        # 读取mat文件
        #print("Reading mat file...")
        f = h5py.File(data_path + data_file, "r")  # 记得更改

        # as it turns out, trying to pickle this is a shit idea :D

        # refs#
        # subsystem#
        # accelData depths images instances labels
        # names namesToIds
        # rawDepthFilenames rawDepths
        # rawRgbFilenames sceneTypes scenes

        rgb_images_fr = np.transpose(f['images'], [0, 2, 3, 1]).astype(np.float32)
        label_images_fr = np.array(f['labels'])

        # 关闭文件
        f.close()
        # 通过在prediction.py文件中将变量维度输出,得知该变量维度为(1449, 640, 480, 3)
        self.rgb_images = rgb_images_fr
        # 通过在prediction.py文件中将变量维度输出,得知该变量维度为(1449, 640, 480)
        self.label_images = label_images_fr

    def __len__(self):
        # 如果是多维矩阵,一般返回第一个维度
        return len(self.rgb_images)  # 1449

    def __getitem__(self, idx):
        # rgb存储rgb图片,维度为(640, 480, 3)
        rgb = self.rgb_images[idx].astype(np.float32)
        # print("rgb.shape:", rgb.shape)  # (640, 480, 3)
        # print(cv2.imread("data/hha/" + str(idx) + ".png", cv2.COLOR_BGR2RGB).shape)#(480, 640, 3)

        # hha存储hha图片,维度为(640, 480, 3)
        hha = np.transpose(cv2.imread("./data/hha/" + str(idx) + ".jpg", cv2.COLOR_BGR2RGB), [1, 0, 2])
        # print("hha.shape:", hha.shape)  # (640, 480, 3)

        # rgb_hha将rgb图片和hha图片进行了合并 维度 (640, 480, 6)
        rgb_hha = np.concatenate([rgb, hha], axis=2).astype(np.float32)
        # print("rgb_hha.shape:", rgb_hha.shape)  # (640, 480, 6)

        # 维度 (640, 480)
        label = self.label_images[idx].astype(np.float32)
        label[label >= 14] = 0

        # 构造和rgb维度相同的零矩阵,因为有0：2,所以维度为 (640, 480, 2)
        xy = np.zeros_like(rgb)[:, :, 0:2].astype(np.float32)

        # random crop
        # random crop 随机裁减 不但提高了模型精度，也增强了模型稳定性,
        if self.crop_type is not None and self.crop_size > 0:
            max_margin = rgb_hha.shape[0] - self.crop_size
            if max_margin == 0:  # crop is original size, so nothing to crop
                self.crop_type = None
            elif self.crop_type == 'Center':
                rgb_hha = rgb[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
                label = label[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2]
                xy = xy[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
            elif self.crop_type == 'Random':
                x_ = np.random.randint(0, max_margin)
                y_ = np.random.randint(0, max_margin)
                rgb_hha = rgb_hha[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
                label = label[y_:y_ + self.crop_size, x_:x_ + self.crop_size]
                xy = xy[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
            else:
                print('Bad crop')  # TODO make this more like, you know, good software
                exit(0)

        # random flip
        # random flip 随机翻转（一行的左右进行翻转），提高模型范化能力
        if self.flip_prob is not None:
            if np.random.random() > self.flip_prob:
                rgb_hha = np.fliplr(rgb_hha).copy()
                label = np.fliplr(label).copy()
                xy = np.fliplr(xy).copy()
        # return 实际上也就确定了数据集的格式 分别是rgb_hha, label, xy数据
        return rgb_hha, label, xy


if __name__ == '__main__':
    data = Dataset(flip_prob=0.5, crop_type='Random', crop_size=0)
    print("data.rgb_images.shape:", data.rgb_images.shape,"data.label_images.shape:",
          data.label_images.shape)  # (1449, 640, 480, 3) (1449, 640, 480)
    print("rgb_hha:", data[0][0].shape)  # (640, 480, 6)
    print("label:", data[0][1].shape)  # (640, 480)
    print("xy:", data[0][2].shape)  # (640, 480, 2)
    print(type(data[0][2]))  # <class 'numpy.ndarray'>
