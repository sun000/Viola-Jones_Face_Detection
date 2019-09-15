from AdaBoost import AdaBoost
from context import Context
import numpy as np
from utils import EPS

class FaceRecognizer(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.adaBoost_models = []

    def train(self):
        while self.ctx.F > self.ctx.F_target:
            adaBoost = AdaBoost(self.ctx)
            adaBoost.train()
            self.adaBoost_models.append(adaBoost)
            #print(self.ctx.valid_predict)
            self.ctx.update_valid_features()
            #print(len(self.ctx.valid_p_features[0]))
            #print(len(self.ctx.valid_n_features[0]))

    def predict(self, test_images):
        predict_positive_idx = list(range(len(test_images)))
        for adaBoost in self.adaBoost_models:
            test_predict = adaBoost.predict(test_images)
            for idx in range(len(test_predict) -1 , -1, -1):
                if test_predict[idx] < 1.0 - EPS:
                    del test_images[idx]
                    del predict_positive_idx[idx]
        predict_labels = []
        for idx in predict_positive_idx:
            while(len(predict_labels)) < idx:
                predict_labels.append(0)
            predict_labels.append(1)
        return predict_labels

def FaceDecetive(object):
    def __init__(self):
        pass

def main():
    from context import get_debug_context
    ctx = get_debug_context()
    faceRecognizer = FaceRecognizer(ctx)
    faceRecognizer.train()
    test_data = []
    #for i in range(100):
        #test_data.append(get_random_sample((24, 24)))
    print(faceRecognizer.predict(ctx.train_p_data + ctx.train_n_data))

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
