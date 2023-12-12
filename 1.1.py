import sys
import cv2
import numpy as np


def main(argv):
    default_file = "coins.jpg"
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print("Error opening image!")
        print("Usage: hough_circle.py [image_name -- default " + default_file + "] \n")
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

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Create a blank image for edge detection
        edges_only = src.copy()

        # Create a copy of the original image for circle centers
        circle_centers_only = np.zeros_like(src)

        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(circle_centers_only, center, 1, (255, 255, 255), 2)

            # Draw circle edges on the edges_only image
            radius = i[2]
            cv2.circle(edges_only, center, radius, (0, 255, 0), 2)

        cv2.imshow("Circle_center", circle_centers_only)
        cv2.imshow("Img_process", edges_only)

    cv2.imshow("Img_src", src)
    cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
