#coding=utf-8
import numpy as np
from utils import poolContext
from utils import EPS
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
    def __init__(self, ctx):
        self.ctx = ctx
        self.last_F, self.last_D = ctx.F, ctx.D
        self.weakclassifiers = []
        self.alpha = [] # 弱分类器的权重
        self.threshold = 0.5
        self.used_features_idx = []
        self.used_features_idx_flag = [0] * len(self.ctx.features_extractors)
        self.used_features_extractors = []
        self.used_valid_features = [] # 验证时不用所有feature所以只加入选中的features
        self.used_train_features = [] # 用于最后更新，train集合


    def train(self): # train_finish用于判断是否训练结束
        self.train_features, self.train_labels = self.ctx.get_train_data()
        self.valid_features, self.valid_labels = self.ctx.get_valid_data()

        self.weights = np.array([1 / ((2 * len(self.ctx.train_n_features[0])) + EPS)] * len(self.ctx.train_p_features[0]) + [1 / ((2 * len(self.ctx.train_p_features[0]))+EPS) ] * len(self.ctx.train_n_features[0]))

        # 训练主要逻辑
        while self.ctx.F > self.last_F * self.ctx.f - EPS:
            self.add_weak_classifier() # 同时计算出对train_data的预测
            self.decrease_threshold() # 降低threshold到符合要求
            self.ctx.F, self.ctx.D = self.evaluate()
            self.ctx.valid_predict = self.predict_from_feature(self.used_valid_features)
            self.ctx.train_predict = self.predict_from_feature(self.used_train_features)
            print("AdaBoost Size ", len(self.weakclassifiers), "F ", self.ctx.F, "D ",
                  self.ctx.D, "last_F ", self.last_F, "last_D ", self.last_D, "threshold ", self.threshold)
        print()


    def add_weak_classifier(self):
        self.weights /= np.sum(self.weights)
        #print("normalize weights ", self.weights)

        # 读取到所有的feature针对每个feature训练一个weakclassifier
        #print("features[0]", features[0])
        with poolContext(processes=16) as pool:
            candidate_classifier = pool.map(partial(Weakclassifier, labels=self.train_labels, weights=self.weights), self.train_features)

        # 选择一个feature
        classifier_idx = range(len(candidate_classifier))
        classifier_idx = sorted(classifier_idx , key = lambda x: candidate_classifier[x].train_error)
        for idx in classifier_idx:
            if self.used_features_idx_flag[idx] == 0:
                self.used_features_idx.append(idx)
                self.used_features_idx_flag[idx] = 1

                self.used_features_extractors.append(self.ctx.features_extractors[idx])
                self.used_valid_features.append(self.valid_features[idx])
                self.used_train_features.append(self.train_features[idx])
                self.weakclassifiers.append(candidate_classifier[idx])

                error = candidate_classifier[idx].train_error
                beta = error / (1.0 - error + EPS)
                train_ouput = candidate_classifier[idx].predict(self.train_features[idx])
                e = np.abs(train_ouput - self.train_labels)
                self.weights *= beta ** (1.0 - e)
                self.alpha.append(np.log(1.0 / (beta + EPS)))
                return
                #print("feature ", self.train_features[idx])
                print("threshold ", candidate_classifier[idx].threshold)
                print("parity ", candidate_classifier[idx].parity)
                print("predict ", train_ouput.astype(int))
                print("labels  ", self.train_labels)
                print("error " , candidate_classifier[idx].train_error)
                print("beta ", beta)
                print("weights ", self.weights)
                return

    def decrease_threshold(self):
        self.threshold = 0.5
        l, r = 0.0, self.threshold
        while r - l > EPS: # 二分找到满足要求的最大threshold
            m = (l + r) / 2.0
            self.threshold = m
            F, D = self.evaluate()
            if D > self.ctx.d * self.last_D - EPS:
                l = m
            else:
                r = m
        self.threshold = max((l + r) / 2.0 - EPS, 0.0)
        #self.ctx.F, self.ctx.D = self.evaluate()

    def evaluate(self): # 应该评估整个级联模型，不只是其中的一个Adaboost,但是这里不能调用上层的函数，所以比较trick的方法是每增加一个级联模型的节点,就删除之前的节点判断的false样本，但是计算fp和dr的时候，正样本使用最初的原始样本大小作为分母
        valid_predict = self.predict_from_feature(self.used_valid_features)

        false_positive = (valid_predict + (1.0 - self.valid_labels) > 2.0 - EPS).astype(float).sum()
        true_positive = ((valid_predict + self.valid_labels) > 2.0 - EPS).astype(float).sum()
        F = false_positive / self.ctx.valid_n_num
        D = true_positive / self.ctx.valid_p_num
        #print("threshold ", self.threshold, "true_positive ", true_positive, "false_positive", false_positive, "all_positive ", self.valid_labels.sum())
        return F, D

    def predict_from_feature(self, features):
        #print(len(features), len(self.weakclassifiers), len(self.alpha))
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
        test_features = self.ctx.get_features_from_images(img, self.used_features_extractors)
        return self.predict_from_feature(test_features)

