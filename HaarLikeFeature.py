import IntegralImage as ii
from enum import Enum
from functools import partial
import numpy as np

class FeatureType(Enum):
    TWO_VERTICAL=(1, 2)
    TWO_HORIZONTAL=(2, 1)
    THREE_VERTICAL=(1, 3)
    THREE_HORIZONTAL=(3, 1)
    FOUR=(2, 2)

FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]

class HaarLikeFeature(object):
    def __init__(self, feature_type, position = (0, 0), size = (0, 0)):
        self.feature_type = feature_type
        self.position = position
        self.size = size

    def get_score(self, int_img):
        a, b = self.position[0]+self.size[0], self.position[1] + self.size[1]
        assert a < int_img.shape[0] and b < int_img.shape[1]
        #print(int_img.shape)
        #print(self.feature_type)
        #return 0.0
        score = 0.0
        if self.feature_type == FeatureType.TWO_VERTICAL:
            left = ii.sum_region(int_img, self.position, (self.size[0], self.size[1] // 2))
            right = ii.sum_region(int_img, (self.position[0], self.position[1] + self.size[1] // 2), (self.size[0], self.size[1] - self.size[1] // 2))
            score = left - right
        elif self.feature_type == FeatureType.TWO_HORIZONTAL:
            top = ii.sum_region(int_img, self.position, (self.size[0] // 2, self.size[1]))
            bottom = ii.sum_region(int_img, (self.position[0] + self.size[0] // 2, self.position[1]), (self.size[0] - self.size[0] // 2, self.size[1]))
            score = top - bottom
        elif self.feature_type == FeatureType.THREE_VERTICAL:
            left = ii.sum_region(int_img, self.position, (self.size[0], self.size[1] // 3))
            mid = ii.sum_region(int_img, (self.position[0], self.position[1] + self.size[1] // 3), (self.size[0], self.size[1] // 3))
            right = ii.sum_region(int_img, (self.position[0], self.position[1] + self.size[1] // 3 * 2), (self.size[0], self.size[1] - self.size[1] // 3 * 2))
            score = mid - left - right
        elif self.feature_type == FeatureType.THREE_HORIZONTAL:
            top = ii.sum_region(int_img, self.position, (self.size[0] // 3, self.size[1]))
            mid = ii.sum_region(int_img, (self.position[0] + self.size[0] // 3, self.position[1]), (self.size[0] // 3, self.size[1]))
            bottom = ii.sum_region(int_img, (self.position[0] + self.size[0] // 3 * 2, self.position[1]), (self.size[0] - self.size[0] // 3 * 2, self.size[1]))
            score = mid - top - bottom
        elif self.feature_type == FeatureType.FOUR:
            left_top = ii.sum_region(int_img, self.position, (self.size[0] // 2, self.size[1] // 2))
            left_botton = ii.sum_region(int_img, (self.position[0] + self.size[0] // 2 , self.position[1]), (self.size[0] - self.size[0] // 2, self.size[1] // 2))
            right_top= ii.sum_region(int_img, (self.position[0], self.position[1] + self.size[1] // 2), (self.size[0], self.size[1] - self.size[1] // 2))
            right_bottom = ii.sum_region(int_img, (self.position[0] + self.size[0] // 2 , self.position[1] + self.size[1] // 2), (self.size[0] - self.size[0] // 2, self.size[1] - self.size[1] // 2))
            score = right_top + left_botton - left_top - right_bottom

        return score

def get_all_feature_extractor(image_size):
    (img_w, img_h) = image_size
    features_extractor = []
    for x in range(img_w):
        for y in range(img_h):
            w = 1
            while w + x < img_w:
                h = 1
                while h + y < img_h:
                    for feature_type in FeatureTypes:
                        if w >= feature_type.value[0] and h >= feature_type.value[1]:
                            features_extractor.append(HaarLikeFeature(feature_type, position=(x, y), size=(w, h)))
                    h += 1
                w += 1
    return features_extractor

def get_features(features_extractor, int_imgs):# int_img array, cal all images' feature in one
    features = list(map(features_extractor.get_score, int_imgs))
    #print(len(features))
    return np.array(features)

def main():
    pass

if __name__ == "__main__":
    main()

