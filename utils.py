#coding=utf-8
import os
import cv2
import multiprocessing
from contextlib import contextmanager

@contextmanager
def poolContext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def read_images(image_dir, image_size=None, normalize=True):
    file_names = os.listdir(image_dir)
    images = []
    for file_name in file_names:
        file_path = image_dir + "/" + file_name
        #print(file_path)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image_size is not None:
            image = cv2.resize(image, image_size)
        if normalize is True:
            cv2.normalize(image, image)
        images.append(image)
    return images

def main():
    images = read_images("data/test_data", (24, 24))
    for image in images:
        print(image.shape)
        #cv2.imshow('', image)
        #cv2.waitKey(0)
    pass

if __name__ == "__main__":
    main()
