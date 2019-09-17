#coding=utf-8
from utils import read_images
from utils import poolContext
from HaarLikeFeature import get_features
from HaarLikeFeature import get_all_feature_extractor
from IntegralImage import to_integral_image
from functools import partial
import numpy as np
from utils import EPS
import random

class Context(object):
    def __init__(self,
                 f,
                 d,
                 F_target,
                 train_p_data_dir,
                 train_n_data_dir,
                 image_size=(24, 24)):
        self.f, self.d = f, d
        self.F, self.D = 1.0, 1.0
        self.F_target = F_target
        self.image_size = image_size
        self.valid_predict = []
        self.train_predict = []

        # 加载数据并提取特征
        print("Load data......")
        self.load_data(train_p_data_dir, train_n_data_dir)
        self.features_extractors = get_all_feature_extractor(image_size)
        print("Extract features......")
        self.extract_features()
        print("Finish extracting features!!!")

    def load_data(self, train_p_data_dir, train_n_data_dir):
        p_data = read_images(train_p_data_dir, self.image_size, normalize=True, limit=5000)
        n_data = read_images(train_n_data_dir, self.image_size, normalize=True, limit=10000)
        random.shuffle(p_data)
        random.shuffle(n_data)
        # 分割出test集合
        split_p = int(len(p_data) * 0.2)
        split_n = int(len(n_data) * 0.2)
        self.test_p_data = p_data[:split_p]
        self.test_n_data = n_data[:split_n]
        p_data = p_data[split_p:]
        n_data = n_data[split_n:]
        split_p = int(len(p_data) * 0.7)
        split_n = int(len(n_data) * 0.7)
        self.train_p_data = p_data[:split_p]
        self.train_n_data = n_data[:split_n]
        self.valid_p_data = p_data[split_p:]
        self.valid_n_data = n_data[split_n:]
        self.train_p_num, self.train_n_num = len(self.train_p_data), len(self.train_n_data)
        self.valid_p_num, self.valid_n_num = len(self.valid_p_data), len(self.valid_n_data)
        # 训练数据自增操作
        self.train_p_data += list(map(np.flipud, self.train_p_data))
        self.train_n_data += list(map(np.flipud, self.train_n_data))
        #self.train_n_data = self.train_n_data[:len(self.train_p_data)]
        print("train data size: ", "p:", len(self.train_p_data), "n:", len(self.train_n_data))
        print("valid data size: ", "p:", len(self.valid_p_data), "n:", len(self.valid_n_data))
        print("test data size: ", "p:", len(self.test_p_data), "n:", len(self.test_n_data))

    def get_features_from_images(self, images, features_extractors=None, int_images=None):
        if features_extractors is None:
            features_extractors = self.features_extractors
        if int_images is None:
            with poolContext(processes=16) as pool:
                int_images = pool.map(to_integral_image, images)
        with poolContext(processes=16) as pool:
            features = pool.map(partial(get_features, int_imgs=int_images), features_extractors)
        return features

    def extract_features(self):
        self.train_p_features = self.get_features_from_images(self.train_p_data)
        print("Finish extracting train_p_feature!!!")
        self.train_n_features = self.get_features_from_images(self.train_n_data)
        print("Finish extracting train_n_feature!!!")
        self.valid_p_features = self.get_features_from_images(self.valid_p_data)
        print("Finish extracting valid_p_feature!!!")
        self.valid_n_features = self.get_features_from_images(self.valid_n_data)
        print("Finish extracting valid_n_feature!!!")

    def get_train_data(self):
        tmp_features = zip(self.train_p_features, self.train_n_features)
        with poolContext(processes=16) as pool:
            train_features = pool.map(np.concatenate, tmp_features)
        labels = np.array([1] * len(self.train_p_features[0]) + [0] * len(self.train_n_features[0]))
        return train_features, labels

    def get_valid_data(self):
        tmp_features = zip(self.valid_p_features, self.valid_n_features)
        with poolContext(processes=16) as pool:
            valid_features = pool.map(np.concatenate, tmp_features)
        labels = np.array([1] * len(self.valid_p_features[0]) + [0] * len(self.valid_n_features[0]))
        return valid_features, labels

    def get_test_data(self):
        labels = np.array([1] * len(self.test_p_data) + [0] * len(self.test_n_data))
        return self.test_p_data + self.test_n_data, labels

    def update_features(self): # 根据预测值更新验证集合，去除正样本中被预测为false的样本，负样本只保留fp样本
        # 更新valid集合，被标记为0的不能进入级联下层的Adaboost
        real_p_num = len(self.valid_p_features[0])
        real_n_num = len(self.valid_n_features[0])
        assert real_p_num + real_n_num == len(self.valid_predict)

        p_need_delete_idx = []
        n_need_delete_idx = []
        for idx in range(real_p_num):
            if self.valid_predict[idx] < EPS:
                p_need_delete_idx.append(idx)
        for idx in range(real_n_num):
            if self.valid_predict[real_p_num + idx] < EPS:
                n_need_delete_idx.append(idx)
        with poolContext(processes=4) as pool:
            self.valid_p_features = pool.map(partial(np.delete, obj=p_need_delete_idx), self.valid_p_features)
        with poolContext(processes=4) as pool:
            self.valid_n_features = pool.map(partial(np.delete, obj=n_need_delete_idx), self.valid_n_features)

        # 更新train集合，被标记为0的不能进入级联下层的Adaboost
        real_p_num = len(self.train_p_features[0])
        real_n_num = len(self.train_n_features[0])
        assert real_p_num + real_n_num == len(self.train_predict)

        n_need_delete_idx = []
        for idx in range(real_n_num):
            if self.train_predict[real_p_num + idx] < EPS:
                n_need_delete_idx.append(idx)
        with poolContext(processes=12) as pool:
            self.train_n_features = pool.map(partial(np.delete, obj=n_need_delete_idx), self.train_n_features)

        self.train_n_num = len(self.train_n_features[0]) # 只更新正样本，valid_feature去除是因为不能通过当前层，但是计算F,D的时候考虑的是所有进入级联模型的样本：w

def get_debug_context():
    ctx = Context(0.5, 0.98, 0.1,
                  #"./data/debug/train_p/",
                  "./data/colorferet/faces/",
                  #"./data/debug/train_n/")
                  "./data/colorferet/non_faces/")
    #ctx.get_train_data()
    #ctx.get_valid_data()
    return ctx

def main():
    get_debug_context()

if __name__ == "__main__":
    main()


