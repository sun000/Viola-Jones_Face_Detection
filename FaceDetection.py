from AdaBoost import AdaBoost

def FaceRecognizer(object):
    def __init__(self, F_target, f, d):
        self.F_target = F_target
        self.f = f
        self.d = d
        adaBoost_models = []

    def train(self, p_img, n_img):
        F, D = 1.0, 1.0
        while F > self.F_target:
            adaBoost = AdaBoost(self.f, self.d, F, D)
            self.adaBoost_models.append(adaBoost)
            F, _ = adaBoost.evaluate()
            n_img = adaBoost.get_false_pos()

    def predict(self):
        pass

def FaceDecetive(object):
    def __init__(self):
        pass
