from utils import read_images
from utils import poolContext
from HaarLikeFeature import get_features
from HaarLikeFeature import get_all_feature_extractor
from IntegralImage import to_integral_image
from functools import partial
import numpy as np
from utils import EPS

class Context(object):
    def __init__(self,
                 f,
                 d,
                 F_target,
                 train_p_data_dir,
                 train_n_data_dir,
                 valid_p_data_dir,
                 valid_n_data_dir,
                 image_size=(24, 24)):
        self.f, self.d = f, d
        self.F, self.D = 1.0, 1.0
        self.F_target = F_target
        self.image_size = image_size

        # 加载数据并提取特征
        print("Load data......")
        self.load_data(train_p_data_dir, train_n_data_dir, valid_p_data_dir, valid_n_data_dir)
        self.valid_predict = []
        self.features_extractors = get_all_feature_extractor(image_size)
        print("Extract features......")
        self.extract_features()
        print("Finish extecting features!")

    def load_data(self, train_p_data_dir, train_n_data_dir, valid_p_data_dir, valid_n_data_dir):
        self.train_p_data = read_images(train_p_data_dir, self.image_size, normalize=True, limit=-1)
        self.train_n_data = read_images(train_n_data_dir, self.image_size, normalize=True, limit=-1)
        self.valid_p_data = read_images(valid_p_data_dir, self.image_size, normalize=True, limit=70)
        self.valid_n_data = read_images(valid_n_data_dir, self.image_size, normalize=True, limit=70)
        self.valid_p_num, self.valid_n_num = len(self.valid_p_data), len(self.valid_n_data)

    def get_features_from_images(self, images, features_extractors=None):
        if features_extractors is None:
            features_extractors = self.features_extractors
        with poolContext(processes=16) as pool:
            int_images = pool.map(to_integral_image, images)
        with poolContext(processes=16) as pool:
            features = pool.map(partial(get_features, int_imgs=int_images), features_extractors)
        return features

    def extract_features(self):
        self.train_features   = self.get_features_from_images(self.train_p_data + self.train_n_data)
        self.valid_p_features = self.get_features_from_images(self.valid_p_data)
        self.valid_n_features = self.get_features_from_images(self.valid_n_data)

    def get_train_data(self):
        labels = np.array([1] * len(self.train_p_data) + [0] * len(self.train_n_data))
        return self.train_features, labels

    def get_valid_data(self):
        tmp_features = zip(self.valid_p_features, self.valid_n_features)
        with poolContext(processes=16) as pool:
            valid_features = pool.map(np.concatenate, tmp_features)
        labels = np.array([1] * len(self.valid_p_features[0]) + [0] * len(self.valid_n_features[0]))
        return valid_features, labels

    def update_valid_features(self): # 根据预测值更新验证集合，去除正样本中被预测为false的样本，负样本只保留fp样本
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

        self.valid_n_num = len(self.valid_n_features[0].shape) # 正样本不需要更新

def get_debug_context():
    ctx = Context(0.3, 0.9, 0.1,
                  "./data/debug/train_p/",
                  "./data/debug/train_n/",
                  "./data/debug/valid_p/",
                  "./data/debug/valid_n/")
    ctx.get_train_data()
    ctx.get_valid_data()
    return ctx

def main():
    get_debug_context()

if __name__ == "__main__":
    main()


