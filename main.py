import cv2
import numpy as np
import sys
from matplotlib import gridspec

# from PyQt5 import QtCore, QtWidgets
# from PyQt5.QtGui import QPixmap
# from HW2UI_ui import Ui_MainWindow
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchsummary import summary
from torchvision.transforms import Compose, ToTensor, Normalize


def load_image():
    global filePath
    try:
        abs_path = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", ".")[0]
        # 获取当前工作目录
        current_dir = os.getcwd()

        # 计算相对路径
        filePath = os.path.relpath(abs_path, current_dir)
    except:
        pass
    # print(abs_path)
    # print(filePath)


def load_image5():
    global filePath
    try:
        abs_path = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", ".")[0]
        # 获取当前工作目录
        current_dir = os.getcwd()

        # 计算相对路径
        filePath = os.path.relpath(abs_path, current_dir)
        # print(filePath)

        image = QPixmap(filePath).scaled(148, 148)
        scene = QtWidgets.QGraphicsScene()  # 加入圖片
        scene.addPixmap(image)  # 將圖片加入 scene
        ui.InferenceImage_View.setScene(scene)
    except:
        pass


def Q1_1():
    def main(argv):
        default_file = "coins.jpg"
        filename = argv[0] if len(argv) > 0 else default_file
        # Loads an image
        src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
        # Check if image is loaded fine
        if src is None:
            print("Error opening image!")
            print(
                "Usage: hough_circle.py [image_name -- default " + default_file + "] \n"
            )
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


def Q1_2():
    # 讀取彩色圖片
    image = cv2.imread(filePath)

    # 將彩色圖片轉成灰階圖片
    I1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 分離通道
    b, g, r = cv2.split(image)

    # 計算I2 = (R + G + B) / 3
    I2 = (r + g + b) / 3

    # 將I2轉換為灰階圖片
    I2 = I2.astype(np.uint8)

    # 顯示灰階圖片
    cv2.imshow("I1", I1)
    cv2.imshow("I2", I2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q1_3():
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


def Q2_1():
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
    hist_equalized_manual, _ = np.histogram(
        equalized_image_manual.flatten(), 256, [0, 256]
    )

    # 繪製圖像和直方圖
    plt.figure(figsize=(16, 8))
    # 使用gridspec調整子圖的高度
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    # 顯示原始圖像和均衡後圖像
    plt.subplot(gs[0, 0]), plt.imshow(image, cmap="gray"), plt.title("Original Image")
    plt.axis("off")  # Remove x and y-axis labels
    plt.subplot(gs[0, 1]), plt.imshow(equalized_image, cmap="gray"), plt.title(
        "Equalized with OpenCV"
    )
    plt.axis("off")  # Remove x and y-axis labels
    plt.subplot(gs[0, 2]), plt.imshow(equalized_image_manual, cmap="gray"), plt.title(
        "Equalized Manually"
    )
    plt.axis("off")  # Remove x and y-axis labels

    # 顯示原始圖像的直方圖
    plt.subplot(gs[1, 0])
    plt.bar(range(256), hist_original, color="steelblue", width=1)
    plt.title("Histogram of Original")
    plt.xlabel("Gray Scale")
    plt.ylabel("Frequency")

    # 顯示均衡後圖像的直方圖(OpenCV)
    plt.subplot(gs[1, 1])
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
    plt.subplot(gs[1, 2])
    plt.bar(range(256), hist_equalized_manual, color="steelblue", width=1)
    plt.title("Histogram of Equalized (Manual)")
    plt.xlabel("Gray Scale")
    plt.ylabel("Frequency")

    plt.tight_layout()
    # 顯示圖片和直方圖
    plt.show()


def Q3_1():
    # Load the image
    image_path = "closing.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Binarize the grayscale image
    threshold = 127
    binary_img = (img > threshold) * 255

    # Step 3: Pad the image with zeros based on the kernel size (K=3)
    pad_size = 1  # Padding size for a 3x3 kernel
    padded_img = np.pad(binary_img, pad_size, mode="constant", constant_values=0)

    # Step 4: Perform dilation using a 3x3 all-ones structuring element
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated_img = np.zeros_like(padded_img)
    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            if np.any(
                padded_img[
                    i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1
                ]
                * kernel_dilate
            ):
                dilated_img[i, j] = 255

    # Step 5: Perform erosion using a 3x3 all-ones structuring element
    kernel_erode = np.ones((3, 3), np.uint8)
    eroded_img = np.copy(dilated_img)
    for i in range(pad_size, dilated_img.shape[0] - pad_size):
        for j in range(pad_size, dilated_img.shape[1] - pad_size):
            if not np.all(
                dilated_img[
                    i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1
                ]
                * kernel_erode
            ):
                eroded_img[i, j] = 0

    # Convert the image to uint8 before displaying
    eroded_img_display = eroded_img.astype(np.uint8)

    # Step 6: Show the image in a popup window
    cv2.imshow("Closing Operation", eroded_img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q3_2():
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
                padded_img[
                    i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1
                ]
                * kernel_erode
            ):
                eroded_img[i, j] = 0

    # Step 5: Perform dilation using a 3x3 all-ones structuring element
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated_img = np.zeros_like(eroded_img)
    for i in range(pad_size, eroded_img.shape[0] - pad_size):
        for j in range(pad_size, eroded_img.shape[1] - pad_size):
            if np.any(
                eroded_img[
                    i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1
                ]
                * kernel_dilate
            ):
                dilated_img[i, j] = 255

    # Convert the image to uint8 before displaying
    dilated_img_display = dilated_img.astype(np.uint8)

    # Step 6: Show the image in a popup window
    cv2.imshow("Opening Operation", dilated_img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q4_1():
    # 建立一個帶有批量歸一化的 VGG19 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg19_bn = models.vgg19_bn(num_classes=10, pretrained=False).to(device)

    # 使用 torchsummary.summary 在終端中顯示模型結構
    summary(vgg19_bn, (3, 32, 32))  # 輸入圖像維度 (3, 224, 224)


def Q5_1():
    # images = [Image.open(image_file) for image_file in filePath2]
    image_folder = "Dataset_OpenCvDl_Hw1/Q5_image/Q5_1"
    # 載入圖像
    image_files = [
        os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
    ]
    images = [Image.open(image_file) for image_file in image_files]

    # 資料增強
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
            transforms.RandomVerticalFlip(),  # 隨機垂直翻轉
            transforms.RandomRotation(30),  # 隨機旋轉 (-30 到 30 度之間)
        ]
    )

    augmented_images = [transform(image) for image in images]

    # 提取檔名（不包含格式）作為標籤
    labels = [os.path.splitext(os.path.basename(file))[0] for file in image_files]

    # 顯示增強後的圖像和標籤在一個新視窗中
    _, axes = plt.subplots(3, 3, figsize=(6, 6))

    for i, (original, augmented, label) in enumerate(
        zip(images, augmented_images, labels)
    ):
        ax = axes[i // 3, i % 3]
        ax.set_title(label)
        ax.imshow(original if i % 3 == 0 else augmented)
        ax.axis("off")

    plt.show()


def Q5_2():
    # 建立一個帶有批量歸一化的 VGG19 模型
    vgg19_bn = models.vgg19_bn(num_classes=10)

    # 使用 torchsummary.summary 在終端中顯示模型結構
    summary(vgg19_bn, (3, 224, 224))  # 輸入圖像維度 (3, 224, 224)


def Q5_3():
    image = cv2.imread("learning_history.jpg")
    # 顯示結果圖像
    cv2.imshow("Training/Validating Accuracy and Loss ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q5_4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [x / 255 for x in [125.3, 23.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # 創建一個一般的 VGG19 模型
    vgg19 = models.vgg19_bn(num_classes=10)

    # 載入你的訓練好的權重
    vgg19.load_state_dict(torch.load("vgg19_final.pt", map_location=device))  # 請確保路徑正確
    vgg19.to(device)

    # 設置模型為評估模式
    vgg19.eval()

    # 載入圖片並進行預處理
    transform = Compose([ToTensor(), Normalize(mean, std)])

    image = Image.open(filePath)
    image = transform(image).unsqueeze(0).to(device)  # 添加一個批次維度並移到GPU（如果可用）

    # 使用模型進行推論
    with torch.no_grad():
        outputs = vgg19(image)

    # 取得類別機率
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # 取得預測的類別索引
    predicted_class = torch.argmax(probabilities).item()

    # 載入CIFAR-10類別名稱對照表
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # 輸出結果
    # print(
    #     "Predicted class: {} ({})".format(class_names[predicted_class], predicted_class)
    # )
    # print("Class probabilities:")
    # for i, prob in enumerate(probabilities):
    #     print("{}: {:.2f}%".format(class_names[i], prob * 100))

    ui.predict_label.setText("Predicted = " + class_names[predicted_class])

    probs = [prob.item() for prob in probabilities]

    # 創建一個長條圖
    plt.figure(figsize=(6, 6))
    plt.bar(class_names, probs, alpha=0.7)

    # 設置圖表標題和軸標籤
    plt.title("Probability of each class")
    plt.xlabel("Class Name")
    plt.ylabel("Probability")

    # 顯示機率值在長條上
    for i, prob in enumerate(probs):
        plt.text(i, prob, f"{prob:.2f}", ha="center", va="bottom")

    # 顯示長條圖
    plt.xticks(rotation=45)  # 使x軸標籤更易讀
    plt.tight_layout()
    plt.show()


app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

ui.LoadImage1_Button.clicked.connect(load_image)
ui.Q1_1_Button.clicked.connect(Q1_1)
ui.Q1_2_Button.clicked.connect(Q1_2)
ui.Q1_3_Button.clicked.connect(Q1_3)
ui.Q2_1_Button.clicked.connect(Q2_1)
ui.Q2_2_Button.clicked.connect(Q2_2)
ui.Q2_3_Button.clicked.connect(Q2_3)
ui.Q3_1_Button.clicked.connect(Q3_1)
ui.Q3_2_Button.clicked.connect(Q3_2)
ui.Q3_3_Button.clicked.connect(Q3_3)
# ui.Q3_4_Button.clicked.connect()
ui.Q4_Button.clicked.connect(Q4)
ui.Q5_Load_Button.clicked.connect(load_image5)
ui.Q5_1_Button.clicked.connect(Q5_1)
ui.Q5_2_Button.clicked.connect(Q5_2)
ui.Q5_3_Button.clicked.connect(Q5_3)
ui.Q5_4_Button.clicked.connect(Q5_4)

MainWindow.show()
app.exec_()
