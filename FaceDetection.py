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
            break

    def predict(self):
        pass

def FaceDecetive(object):
    def __init__(self):
        pass

def main():
    from context import get_debug_context
    ctx = get_debug_context()
    faceRecognizer = FaceRecognizer(ctx)
    faceRecognizer.train()

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
