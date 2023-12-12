import sys
import cv2
import numpy as np


def count_coins(image_path):
    # Loads an image
    src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Check if the image is loaded fine
    if src is None:
        print("Error opening image!")
        return -1

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        rows / 16,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=40,
    )
    circles = np.uint16(np.around(circles))
    coin_count = len(circles[0])
    print("Number of coins:", coin_count)

    return 0


if __name__ == "__main__":
    image_path = "coins.jpg"
    count_coins(image_path)
