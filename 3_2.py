import cv2
import numpy as np

# Load the image
image_path = "opening.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 2: Binarize the grayscale image
threshold = 127
binary_img = (img > threshold) * 255

# Step 3: Pad the image with zeros based on the kernel size (K=3)
pad_size = 1  # Padding size for a 3x3 kernel
padded_img = np.pad(binary_img, pad_size, mode="constant", constant_values=0)

# Step 4: Perform erosion using a 3x3 all-ones structuring element
kernel_erode = np.ones((3, 3), np.uint8)
eroded_img = np.copy(padded_img)
for i in range(pad_size, padded_img.shape[0] - pad_size):
    for j in range(pad_size, padded_img.shape[1] - pad_size):
        if not np.all(
            padded_img[i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1]
            * kernel_erode
        ):
            eroded_img[i, j] = 0

# Step 5: Perform dilation using a 3x3 all-ones structuring element
kernel_dilate = np.ones((3, 3), np.uint8)
dilated_img = np.zeros_like(eroded_img)
for i in range(pad_size, eroded_img.shape[0] - pad_size):
    for j in range(pad_size, eroded_img.shape[1] - pad_size):
        if np.any(
            eroded_img[i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1]
            * kernel_dilate
        ):
            dilated_img[i, j] = 255

# Convert the image to uint8 before displaying
dilated_img_display = dilated_img.astype(np.uint8)

# Step 6: Show the image in a popup window
cv2.imshow("Opening Operation", dilated_img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
