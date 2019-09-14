from AdaBoost import AdaBoost
import numpy as np
from utils import EPS
from utils import read_images

class FaceRecognizer(object):
    def __init__(self, F_target, f, d):
        self.F_target = F_target
        self.f, self.d = f, d
        self.adaBoost_models = []

    def train(self, p_img, n_img, valid_p_img, valid_n_img):
        F, D = 1.0, 1.0
        while F > self.F_target:
            adaBoost = AdaBoost(self.f, self.d, F, D)
            adaBoost.train(p_img, n_img, valid_p_img, valid_n_img)
            self.adaBoost_models.append(adaBoost)
            F, D = adaBoost.F, adaBoost.D

            valid_n_predict = adaBoost.predict(valid_n_img)
            #n_img = 
            #print(valid_n_predict.shape)
            #print(type(valid_n_predict))
            #print("it's end")

    def predict(self):
        pass

def FaceDecetive(object):
    def __init__(self):
        pass

def main():
    #p_img = []
    p_img = read_images("./data/colorferet/face_data", image_size=(24,24), normalize=True)[100:]
    n_img = []
    #valid_p_img = []
    valid_p_img = p_img[:20]
    p_img = p_img[20:]
    valid_n_img = []
    #for i in range(41):
        #p_img.append(get_random_sample((24, 24)))
    for i in range(53):
        n_img.append(get_random_sample((24, 24)))
    #for i in range(100):
    #    valid_p_img.append(get_random_sample((24, 24)))
    for i in range(50):
        valid_n_img.append(get_random_sample((24, 24)))

    faceRecognizer = FaceRecognizer(F_target=0.1, f=0.3, d=0.9)
    faceRecognizer.train(p_img, n_img, valid_p_img, valid_n_img)


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
