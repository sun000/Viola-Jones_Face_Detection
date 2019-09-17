#coding=utf-8
from AdaBoost import AdaBoost
from context import Context
import numpy as np
from utils import EPS
import time

class FaceRecognizer(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.adaBoost_models = []

    def train(self):
        while self.ctx.F > self.ctx.F_target - EPS:
            adaBoost = AdaBoost(self.ctx)
            adaBoost.train()
            self.adaBoost_models.append(adaBoost)
            #print(self.ctx.valid_predict)
            self.ctx.update_features()
            #print(len(self.ctx.valid_p_features[0]))
            #print(len(self.ctx.valid_n_features[0]))

    def predict(self, test_images):
        predict_positive_idx = list(range(len(test_images) + 1))
        for adaBoost in self.adaBoost_models:
            test_predict = adaBoost.predict(test_images)
            #print(np.array(test_predict))
            for idx in range(len(test_predict) -1 , -1, -1):
                #print(test_predict[idx], EPS, test_predict[idx] < EPS)
                if test_predict[idx] < EPS:
                    del test_images[idx]
                    del predict_positive_idx[idx]
        predict_labels = []
        #print(np.array(predict_positive_idx))
        for idx in predict_positive_idx:
            while(len(predict_labels)) < idx:
                predict_labels.append(0)
            predict_labels.append(1)
        return np.array(predict_labels[:-1])

def FaceDecetive(object):
    def __init__(self):
        pass

def main():
    from context import get_debug_context
    start_time = time.time()
    ctx = get_debug_context()
    faceRecognizer = FaceRecognizer(ctx)
    ctx_end_time = time.time()
    print("Training start......")
    faceRecognizer.train()
    train_end_time = time.time()
    print("Training end!!!")
    print("Testing......")
    test_data, test_labels = ctx.get_test_data()
    test_predict = faceRecognizer.predict(test_data)

    false_poeitive = (test_predict + (1.0 - test_labels) > 2.0 - EPS).astype(float).sum()
    true_positive = ((test_predict + test_labels) > 2.0 - EPS).astype(float).sum()
    F = false_poeitive / len(ctx.test_n_data)
    D = true_positive / len(ctx.test_p_data)
    test_end_time = time.time()
    print("F ", F, "D", D)
    print("build context time: %s s" % (ctx_end_time - start_time))
    print("training time: %s s" % (train_end_time - ctx_end_time))
    print("testing time: %s s" % (test_end_time - train_end_time))

def get_random_sample(size):
    sample = np.random.randint(0, 255, size=size).astype(float)
    sample -= np.mean(sample)
    std = np.std(sample)
    if std > EPS or std < -EPS:
        sample /= std
    #print(sample.shape)
    return sample

if __name__ == "__main__":
    main()
