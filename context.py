from utils import read_images
from utils import poolContext
from HaarLikeFeature import get_features
from HaarLikeFeature import get_all_feature_extractor
from IntegralImage import to_integral_image
from functools import partial
import numpy as np

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
        self.features_extractors = get_all_feature_extractor(image_size)
        print("Extract features......")
        self.extract_features()
        print("Finish extecting features!")

    def load_data(self, train_p_data_dir, train_n_data_dir, valid_p_data_dir, valid_n_data_dir):
        self.train_p_data = read_images(train_p_data_dir, self.image_size, normalize=True, limit=30)
        self.train_n_data = read_images(train_n_data_dir, self.image_size, normalize=True, limit=30)
        self.valid_p_data = read_images(valid_p_data_dir, self.image_size, normalize=True, limit=10)
        self.valid_n_data = read_images(valid_n_data_dir, self.image_size, normalize=True, limit=10)

    def get_features_from_images(self, images):
        with poolContext(processes=16) as pool:
            int_images = pool.map(to_integral_image, images)
        with poolContext(processes=16) as pool:
            features = pool.map(partial(get_features, int_imgs=int_images), self.features_extractors)
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
        labels = np.array([1] * len(self.valid_p_data) + [0] * len(self.valid_n_data))
        return valid_features, labels

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


