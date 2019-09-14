#coding=utf-8
import numpy as np
from HaarLikeFeature import get_all_feature_extractor
from HaarLikeFeature import get_features
from utils import read_images
from utils import poolContext
from utils import EPS
from IntegralImage import to_integral_image
from functools import partial

class Weakclassifier(object):
    def __init__(self, inputs, labels, weights):
        self.parity = 1.0 # 控制方向
        self.threshold = 0.0
        self.train_error = self.train(inputs, labels, weights)

    def train(self, inputs, labels, weights): # 训练
        assert len(labels) == len(inputs)
        indexs = range(len(labels))
        indexs = sorted(indexs, key=lambda x: inputs[x])
        p_weight_sum, n_weight_sum = 0.0, 0.0
        p_weight_tot, n_weight_tot = 0.0, 0.0
        for i in range(len(labels)):
            if labels[i] == 1:
                p_weight_tot += weights[i]
            else:
                n_weight_tot += weights[i]

        error = 1.0
        for i in range(len(indexs)):
            index = indexs[i]

            if int(labels[index]) == 1:
                p_weight_sum += weights[index]
            else:
                n_weight_sum += weights[index]

            if p_weight_sum  + (n_weight_tot - n_weight_sum) < error:
                error = p_weight_sum + (n_weight_tot - n_weight_sum)
                self.threshold = inputs[index] if i == len(indexs) - 1 else (inputs[indexs[i]] + inputs[indexs[i + 1]]) / 2.0
                self.parity = -1.0
            elif n_weight_sum + (p_weight_tot - p_weight_sum) < error:
                self.threshold = inputs[index] if i == len(indexs) - 1 else (inputs[indexs[i]] + inputs[indexs[i + 1]]) / 2.0
                error = n_weight_sum + (p_weight_tot - p_weight_sum)
                self.parity = 1.0
        return error

    def predict(self, feature):
        return (feature * self.parity - EPS < self.threshold * self.parity).astype(float)

class AdaBoost(object):
    def __init__(self, f, d, last_F, last_D): # 最大可接受的fp rate和最小可接受的detection rate
        self.weakclassifiers = []
        self.used_features_extractor = [] #每个若分类其所对应的特征类型（下标）
        self.alpha = [] # 弱分类器的权重
        #self.weakPredict = [] # 弱分类器对每个样本的预测
        self.f, self.d = f, d
        self.F, self.D = last_F, last_D
        self.last_F, self.last_D = last_F, last_D
        self.threshold = 0.5
        self.weights = None
        self.train_labels = None
        self.feature_extractors = None
        self.used_features_idx = None # 已经使用过的feature不能重复使用

    def train(self, p_img, n_img, valid_p_img, valid_n_img): # train_finish用于判断是否训练结束
        assert len(n_img) == 0 or p_img[0].shape == n_img[0].shape

        self.weights = np.array([1 / (2 * len(n_img))] * len(p_img) + [1 / (2 * len(p_img))] * len(n_img))
        #print("weight: ", self.weights.shape, "\n", self.weights)
        self.train_labels = np.array([1] * len(p_img) + [0] * len(n_img))
        self.valid_labels = np.array([1] * len(valid_p_img) + [0] * len(valid_n_img))
        #print("labels ", self.train_labels.shape, self.valid_labels.shape)

        # get all features
        self.feature_extractors = get_all_feature_extractor(p_img[0].shape)
        self.used_features_idx = [0] * len(self.feature_extractors)

        with poolContext(processes=4) as pool:
            train_int_imgs = pool.map(to_integral_image, p_img + n_img)
        with poolContext(processes=16) as pool:
            self.train_features = pool.map(partial(get_features, int_imgs=train_int_imgs), self.feature_extractors)

        with poolContext(processes=4) as pool:
            self.valid_int_imgs = pool.map(to_integral_image, valid_p_img + valid_n_img)

        self.valid_features = [] # 验证集的特征根据选择的特征来计算，不用计算所有特征

        # 训练主要逻辑
        while self.F > self.last_F * self.f - EPS:
            self.add_weak_classifier() # 同时计算出对train_data的预测
            self.decrease_threshold() # 降低threshold到符合要求
            #self.last_F, self.last_D = self.F, self.D
            self.F, self.D = self.evaluate()
            print("AdaBoost Size ", len(self.weakclassifiers), "F ", self.F, "D ", self.D, "last_F ", self.last_F, "last_D ", self.last_D)

        print()


    def add_weak_classifier(self):

        self.weights /= np.sum(self.weights)
        #print("normalize weights ", self.weights)

        # 读取到所有的feature针对每个feature训练一个weakclassifier
        #print("features[0]", features[0])
        with poolContext(processes=16) as pool:
            candidate_classifier = pool.map(partial(Weakclassifier, labels=self.train_labels, weights=self.weights), self.train_features)

        #print("weights ", self.weights)

        # 选择一个feature
        classifier_idx = range(len(candidate_classifier))
        classifier_idx = sorted(classifier_idx , key = lambda x: candidate_classifier[x].train_error)
        for idx in classifier_idx:
            if self.used_features_idx[idx] == 0:
                self.used_features_idx[idx] = 1
                self.weakclassifiers.append(candidate_classifier[idx])
                self.used_features_extractor.append(self.feature_extractors[idx])
                self.valid_features.append(get_features(self.feature_extractors[idx], self.valid_int_imgs))

                error = candidate_classifier[idx].train_error
                beta = error / (1.0 - error)
                train_ouput = candidate_classifier[idx].predict(self.train_features[idx])
                e = np.abs(train_ouput - self.train_labels)
                self.weights *= beta ** (1.0 - e)
                self.alpha.append(np.log(1.0 / (beta + EPS)))
                #return
                #print("feature ", self.train_features[idx])
                #print("threshold ", candidate_classifier[idx].threshold)
                #print("parity ", candidate_classifier[idx].parity)
                #print("predict ", train_ouput)
                #print("labels  ", self.train_labels)
                #print("error " , candidate_classifier[idx].train_error)
                #print("beta ", beta)
                #print("weights ", self.weights)
                return

    def decrease_threshold(self):
        self.threshold = 0.5
        l, r = 0.0, self.threshold
        while r - l > EPS: # 二分找到满足要求的最大threshold
            m = (l + r) / 2.0
            self.threshold = m
            F, D = self.evaluate()
            #print("m ", m, "F ", F, "D ", D)
            if D > self.d * self.last_D:
                l = m
            else:
                r = m
        self.threshold = max((l + r) / 2.0 - EPS, 0.0)
        self.F, self.D = self.evaluate()

    def evaluate(self):
        valid_predict = self.predict_from_feature(self.valid_features)
        false_positive = (valid_predict + (1 - self.valid_labels) > 2.0 - EPS).astype(float).sum()
        true_positive = ((valid_predict + self.valid_labels) > 2.0 - EPS).astype(float).sum()
        F = false_positive / (1 - self.valid_labels).sum()
        D = true_positive / self.valid_labels.sum()
        #print("threshold ", self.threshold, "true_positive ", true_positive, "all_positive ", self.valid_labels.sum())
        return F, D

    def predict_from_feature(self, features):
        assert len(features) == len(self.weakclassifiers) and len(features) == len(self.alpha) and len(features) > 0
        predict_score = np.zeros(len(features[0]))

        alpha_sum = 0.0
        for (feature, classifier, alpha_now) in zip(features, self.weakclassifiers, self.alpha):
            h = classifier.predict(feature)
            predict_score += h * alpha_now
            alpha_sum += alpha_now
        #print(predict_score)
        predict_labels = (predict_score >= self.threshold * alpha_sum - EPS).astype(float)
        return predict_labels

    def predict(self, img):
        with poolContext(processes=4) as pool:
            test_int_imgs = pool.map(to_integral_image, img)
        with poolContext(processes=16) as pool:
            test_features = pool.map(partial(get_features, int_imgs=test_int_imgs), self.used_features_extractor)
        #print("weakClassifer", len(self.weakclassifiers), "used_features ", test_features)
        return self.predict_from_feature(test_features)


def main():
    adaBoost = AdaBoost(f=0.3, d=0.9, last_F=1.0, last_D=1.0)
    #p_img = read_images("data/test_data", image_size=(24, 24), normalize=True)
    #n_img = [np.zeros(p_img[0].shape)] * len(p_img)
    p_img = []
    n_img = []
    valid_p_img = []
    valid_n_img = []
    for i in range(41):
        p_img.append(get_random_sample((24, 24)))
    for i in range(93):
        n_img.append(get_random_sample((24, 24)))
    for i in range(100):
        valid_p_img.append(get_random_sample((24, 24)))
    for i in range(200):
        valid_n_img.append(get_random_sample((24, 24)))

    adaBoost.train(p_img, n_img, valid_p_img, valid_n_img)

def get_random_sample(size):
    sample = np.random.normal(size=size)
    sample -= np.mean(sample)
    std = np.std(sample)
    if std > EPS or std < -EPS:
        sample /= std
    #print(sample.shape)
    return sample

if __name__ == "__main__":
    main()
