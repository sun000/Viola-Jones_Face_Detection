import numpy as np

def to_integral_image(img):
    (w, h) = img.shape
    integral_img = np.zeros((w + 1, h + 1))
    for x in range(w):
        for y in range(h):
            integral_img[x + 1][y + 1] = integral_img[x][y + 1] + integral_img[x + 1][y] - integral_img[x][y] + img[x][y] #容斥定理
    return integral_img

def sum_region(integral_img, position = (0, 0), size = (0, 0)):
    top_left = position
    bottom_right = (top_left[0] + size[0], top_left[1] + size[1])
    return integral_img[bottom_right[0]][bottom_right[1]] - integral_img[bottom_right[0]][top_left[1]] + integral_img[top_left[0]][top_left[1]] - integral_img[top_left[0]][bottom_right[1]]

def main():
    img = np.ones((3, 3))
    integral_img = to_integral_image(img)
    print(img)
    print(integral_img)
    print(sum_region(integral_img, (0, 0), (3, 3)))
    print(sum_region(integral_img, (0, 0), (2, 2)))
    print(sum_region(integral_img, (0, 0), (1, 2)))
    print(sum_region(integral_img, (1, 1), (2, 2)))
    print(sum_region(integral_img, (1, 2), (1, 1)))
    print(sum_region(integral_img, (2, 2), (1, 1)))
    print(sum_region(integral_img, (2, 2), (0, 0)))


if __name__ == "__main__":
    main()
