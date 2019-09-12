import numpy as np
from HaarLikeFeature import HaarLikeFeature
from HaarLikeFeature import FeatureTypes

class WeakClassifer(object):
    def __init__(self, inputs, labels, weights):
        self.parity = 1.0 # 控制方向
        self.threshold = 0.0
        self.train_error = self.train(inputs, labels, weights)

    def train(self, inputs, labels, weights): # 训练
        indexs = range(len(labels))
        indexs = sorted(indexs, key=lambda x: inputs[x])
        p_weight_sum, n_weight_sum = 0.0, 0.0
        p_weight_tot, n_weight_tot = 0.0, 0.0
        for i in range(len(labels)):
            if labels[i] > 0:
                p_weight_tot += weights[i]
            else:
                n_weight_tot += weights[i]

        error = 1.0
        for i in range(len(indexs)):
            index = indexs[i]
            self.threshold = inputs[index] if i == len(inputs) - 1 else (inputs[indexs[i]] + inputs[indexs[i + 1]]) / 2.0
            if labels[index] == 1:
                p_weight_sum += weights[index]
            else:
                n_weight_sum += weights[index]

            if p_weight_sum  + (n_weight_tot - n_weight_sum) < error:
                error = p_weight_sum  + (n_weight_tot - n_weight_sum)
                self.parity = -1.0
            elif n_weight_sum + (p_weight_tot - p_weight_sum) < error:
                error = n_weight_sum + (p_weight_tot - p_weight_sum)
                self.parity = 1.0
        return error

    def predict(self, feature):
        return 1 if feature * self.parity < self.threshold * self.parity else 0

class AdaBoost(object):
    def __init__(self, f, d, last_F, last_D): # 最大可接受的fp rate和最小可接受的detection rate
        self.weakClassifers = []
        self.alpha = [] # 弱分类器的权重
        self.weakPredict = [] # 弱分类器对每个样本的预测
        self.f = f
        self.d = d
        self.F = last_F
        self.D = last_D
        self.last_F = last_F
        self.last_D = last_D
        self.threshold = 0.5

    def train(self, p_img, n_img, learning_rate): # train_finish用于判断是否训练结束
        while self.F > self.last_F * self.f:
           self.add_weak_classifer(p_img, n_img) # 同时计算出对train_data的预测
           self.decrease_threshold() # 降低threshold到符合要求
           self.F, self.D = self.evaluate()

    def add_weak_classifer(self, p_img, n_img):
        pass

    def decrease_threshold(self):
        l, r = 0.0, self.threshold
        while r - l > 0.05: # 二分找到满足要求的最大threshold
            m = (l + r) / 2.0
            _, D = self.evaluate(threshold=m)
            if D > self.d * self.last_D:
                l = m
            else:
                r = m
        self.threshold = (l + r) / 2.0

    def evaluate(self, threshold=None):
        if threshold is None:
            threshold = self.threshold

        F, D = 0.0, 0.0
        return F, D

    def get_false_pos(self):
        pass

    def predict(self, img):
        pass

