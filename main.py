import cv2
import random
import numpy as np


def smash_and_sample_random_parts(image, part_size, num_samples):
    parts = []
    for i in range(num_samples):
        start_x = random.randint(0, image.shape[0] - part_size[0])
        start_y = random.randint(0, image.shape[1] - part_size[1])
        parts.append(image[
                     start_x: start_x + part_size[0],
                     start_y: start_y + part_size[1],
                     :])
    return parts


def do_sliding_window_thing(image, window):
    # we can slide a window over the image like this
    # NOTE: we could just use cv2.FILTER, but this will be more flexible
    # First, create an output array
    output = image * 0
    # num pixels under filter
    num = window.shape[0] * window.shape[1]
    for x in range(0, image.shape[0] - window.shape[0]):
        for y in range(0, image.shape[1] - window.shape[1]):
            output[x, y] = int(np.sum(np.multiply(image[x: x + window.shape[0], y: y + window.shape[1]], window)) / num)
        if x % 50 == 0:
            print(f"DONE {x} / {image.shape[0]} rows")
    return output


FILTER = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1]
])

image = cv2.imread("pelda.jpeg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered_image = do_sliding_window_thing(gray_image, FILTER)

cv2.namedWindow("imwindow")

cv2.imshow("imwindow", gray_image)
cv2.waitKey(2000)
cv2.imshow("imwindow", filtered_image)
cv2.waitKey(2000)

parts = smash_and_sample_random_parts(image, np.array([150, 150]), 10)
for part in parts:
    cv2.imshow("imwindow", part)
    cv2.waitKey(2000)

# here we could do a gaussian blur (but maybe not for this app)
# I don't remember OTSU algorithm, but the 2 numerical params have some influence...
threshold, binary_image = cv2.threshold(gray_image, 0, 218, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("imwindow", binary_image)

cv2.waitKey(0)