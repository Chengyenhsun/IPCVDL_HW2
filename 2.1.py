import cv2
import numpy as np
import matplotlib.pyplot as plt

# 載入灰度圖像
image_path = "histoEqualGray2.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 進行OpenCV的直方圖均衡
equalized_image = cv2.equalizeHist(image)

# 計算原始圖像和均衡後圖像的直方圖
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# 繪製圖像和直方圖
plt.figure(figsize=(12, 8))

# 顯示原始圖像和均衡後圖像
plt.subplot(2, 2, 1), plt.imshow(image, cmap="gray"), plt.title("Original Image")
plt.subplot(2, 2, 2), plt.imshow(equalized_image, cmap="gray"), plt.title(
    "Equalized with OpenCV"
)

# 顯示原始圖像的直方圖
plt.subplot(2, 2, 3)
plt.bar(range(256), hist_original[:, 0], color="steelblue", width=1)
plt.title("Histogram of Original")

# 顯示均衡後圖像的直方圖
plt.subplot(2, 2, 4)
plt.bar(range(256), hist_equalized[:, 0], color="steelblue", width=1)
plt.title("Histogram of Equalized(OpenCV)")

# 顯示圖片和直方圖
plt.show()
