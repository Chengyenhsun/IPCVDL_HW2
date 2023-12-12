import cv2
import numpy as np
import matplotlib.pyplot as plt

# 載入灰度圖像
image_path = "histoEqualGray2.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 進行OpenCV的直方圖均衡
equalized_image = cv2.equalizeHist(image)

# 計算原始圖像的直方圖
hist_original, bins = np.histogram(image.flatten(), 256, [0, 256])

# 計算PDF
pdf = hist_original / np.sum(hist_original)

# 計算CDF
cdf = np.cumsum(pdf)

# 創建均衡化的查找表
cdf_normalized = (cdf * 255).astype("uint8")

# 使用查找表進行均衡化
equalized_image_manual = cv2.LUT(image, cdf_normalized)

# 計算均衡後圖像的直方圖
hist_equalized_manual, _ = np.histogram(equalized_image_manual.flatten(), 256, [0, 256])

# 繪製圖像和直方圖
plt.figure(figsize=(16, 8))

# 顯示原始圖像和均衡後圖像
plt.subplot(2, 3, 1), plt.imshow(image, cmap="gray"), plt.title("Original Image")
plt.subplot(2, 3, 2), plt.imshow(equalized_image, cmap="gray"), plt.title(
    "Equalized with OpenCV"
)
plt.subplot(2, 3, 3), plt.imshow(equalized_image_manual, cmap="gray"), plt.title(
    "Equalized Manually"
)

# 顯示原始圖像的直方圖
plt.subplot(2, 3, 4)
plt.bar(range(256), hist_original, color="steelblue", width=1)
plt.title("Histogram of Original")
plt.xlabel("Gray Scale")
plt.ylabel("Frequency")

# 顯示均衡後圖像的直方圖(OpenCV)
plt.subplot(2, 3, 5)
plt.bar(
    range(256),
    cv2.calcHist([equalized_image], [0], None, [256], [0, 256])[:, 0],
    color="steelblue",
    width=1,
)
plt.title("Histogram of Equalized (OpenCV)")
plt.xlabel("Gray Scale")
plt.ylabel("Frequency")

# 顯示均衡後圖像的直方圖(Manual)
plt.subplot(2, 3, 6)
plt.bar(range(256), hist_equalized_manual, color="steelblue", width=1)
plt.title("Histogram of Equalized (Manual)")
plt.xlabel("Gray Scale")
plt.ylabel("Frequency")

# 顯示圖片和直方圖
plt.show()
