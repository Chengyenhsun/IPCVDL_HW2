import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torchsummary import summary
from torchvision.transforms import transforms, Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPathItem,
    QWidget,
)
from hw2_ui import Ui_MainWindow


class DrawingScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(DrawingScene, self).__init__(parent)
        self.pos_xy = []
        self.captured_image = None

    def mousePressEvent(self, event):
        pos_tmp = event.scenePos()
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseMoveEvent(self, event):
        pos_tmp = event.scenePos()
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = QtCore.QPointF(-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255), 15, QtCore.Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == QtCore.QPointF(-1, -1):
                    point_start = QtCore.QPointF(-1, -1)
                    continue
                if point_start == QtCore.QPointF(-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start, point_end)
                point_start = point_end


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

        image = QPixmap(filePath).scaled(248, 263)
        scene = QtWidgets.QGraphicsScene()  # 加入圖片
        scene.addPixmap(image)  # 將圖片加入 scene
        ui.Q5_graphicview.setScene(scene)
    except:
        pass


def Q1_1():
    image = cv2.imread(filePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        edges_only = image.copy()

        # Create a copy of the original image for circle centers
        circle_centers_only = np.zeros_like(image)

        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(circle_centers_only, center, 1, (255, 255, 255), 2)

            # Draw circle edges on the edges_only image
            radius = i[2]
            cv2.circle(edges_only, center, radius, (0, 255, 0), 2)

        cv2.imshow("Circle_center", circle_centers_only)
        cv2.imshow("Img_process", edges_only)

    cv2.imshow("Img_src", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Q1_2():
    image = cv2.imread(filePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    ui.Q1coins.setText("There are " + str(coin_count) + " coins in the image.")


def Q2_1():
    image = cv2.imread(filePath)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 進行OpenCV的直方圖均衡
    equalized_image = cv2.equalizeHist(img)
    # 計算原始圖像的直方圖
    hist_original, bins = np.histogram(img.flatten(), 256, [0, 256])
    # 計算PDF
    pdf = hist_original / np.sum(hist_original)
    # 計算CDF
    cdf = np.cumsum(pdf)
    # 創建均衡化的查找表
    cdf_normalized = (cdf * 255).astype("uint8")
    # 使用查找表進行均衡化
    equalized_image_manual = cv2.LUT(img, cdf_normalized)
    # 計算均衡後圖像的直方圖
    hist_equalized_manual, _ = np.histogram(
        equalized_image_manual.flatten(), 256, [0, 256]
    )

    # 繪製圖像和直方圖
    plt.figure(figsize=(16, 8))
    # 使用gridspec調整子圖的高度
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
    # 顯示原始圖像和均衡後圖像
    plt.subplot(gs[0, 0]), plt.imshow(img, cmap="gray"), plt.title("Original Image")
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
    image = cv2.imread(filePath)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    image = cv2.imread(filePath)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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


def Q4_2():
    image_path = "learning_history.jpg"
    # image = cv2.imread(image_path)
    image = QPixmap(image_path).scaled(440, 260)
    scene = QtWidgets.QGraphicsScene()  # 加入圖片
    scene.addPixmap(image)  # 將圖片加入 scene
    ui.Q4_graphicview.setScene(scene)


def Q4_3():
    # 獲取 QGraphicsView 的大小
    view_size = ui.Q4_graphicview.viewport().size()

    # 創建 QImage 並設置大小
    image = QtGui.QImage(view_size, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("black"))  # 將背景填充為白色

    # 創建 QPainter 以在 QImage 上繪製
    painter = QtGui.QPainter(image)
    ui.Q4_graphicview.render(painter)
    painter.end()

    # 保存圖片
    image.save("image.jpg")
    print("Image saved as image.jpg")

    image_path = "image.jpg"
    # 載入擷取的畫面
    image = Image.open(image_path).convert("L")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.1307]
    std = [0.3081]

    # 創建一個一般的 VGG19 模型
    vgg19_bn = models.vgg19_bn(num_classes=10)
    # Modify the first layer to accept a single channel input
    vgg19_bn.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    # Remove the last max pooling layer to better suit the smaller input size
    vgg19_bn.features = vgg19_bn.features[:-1]
    # 載入你的訓練好的權重
    vgg19_bn.load_state_dict(
        torch.load("vgg19_final.pt", map_location=device)
    )  # 請確保路徑正確
    vgg19 = vgg19_bn.to(device)

    # 設置模型為評估模式
    vgg19.eval()
    # 載入圖片並進行預處理
    transform = Compose([Resize((32, 32)), ToTensor(), Normalize(mean, std)])

    image = transform(image).unsqueeze(0).to(device)  # 添加一個批次維度並移到GPU（如果可用）
    # 使用模型進行推論
    with torch.no_grad():
        outputs = vgg19(image)
    # 取得類別機率
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    print(probabilities)
    # 取得預測的類別索引
    predicted_class = torch.argmax(probabilities).item()
    # 載入類別名稱對照表
    class_names = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    ui.Q4_predict.setText("predict = " + class_names[predicted_class])

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


def Q4_4():
    scene = ui.Q4_graphicview.scene()

    # Clear the drawn lines by clearing the pos_xy attribute
    scene.pos_xy = []

    # Update the scene to reflect the changes
    scene.update()

    # 清除場景中的所有項目
    scene.clear()


def Q5_1():
    # 定義自定義資料集類別
    class CustomDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.data = datasets.ImageFolder(root=root_dir, transform=transform)
            self.classes = self.data.classes

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # 設定資料集路徑
    dataset_path = "dataset/inference_dataset"
    inference_dataset = CustomDataset(
        root_dir=dataset_path,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    )

    # 創建 DataLoader
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=True)

    # 取得一張每個類別的圖片
    class_images = {}
    for batch in inference_dataloader:
        image, label = batch
        class_name = inference_dataset.classes[label.item()]

        # 存儲每個類別的圖片
        if class_name not in class_images:
            class_images[class_name] = image

        # 當每個類別都取了一張圖片後，顯示並退出迴圈
        if len(class_images) == len(inference_dataset.classes):
            break

    # 顯示兩張圖片
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, (class_name, image) in enumerate(class_images.items()):
        axes[i].imshow(image.squeeze().permute(1, 2, 0).numpy())
        axes[i].set_title(f"Class: {class_name}")
        axes[i].axis("off")  # 關閉 x 和 y 軸

    plt.show()


def Q5_2():
    # Build ResNet50 model
    resnet50 = models.resnet50(weights=None)
    num_ftrs = resnet50.fc.in_features
    # print(num_ftrs)
    # Replace the output layer with a FC layer of 1 node and Sigmoid activation
    resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50 = resnet50.to(device)

    # Display the model structure using torchsummary
    summary(resnet50, (3, 224, 224))  # Input image dimensions: (3, 224, 224)


def Q5_3():
    image_path = "Accuracy_Comparison.jpg"
    image = cv2.imread(image_path)

    cv2.imshow("Accuracy_Comparison", image)
    cv2.waitKey(0)  # 顯示圖片並等待任意按鍵關閉視窗
    cv2.destroyAllWindows()


def Q5_4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ResNet50-specific mean and std values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Create a ResNet50 model
    resnet50 = models.resnet50(pretrained=False, num_classes=2)
    # Load pre-trained weights
    resnet50.load_state_dict(
        torch.load(
            "resnet50/with_RE/2resnet50_checkpoint_epoch_20.pt", map_location=device
        )
    )  # Make sure the path is correct
    resnet50.to(device)
    resnet50.eval()
    # Load and preprocess the input image
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    image = Image.open(filePath)
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference using the model
    with torch.no_grad():
        outputs = resnet50(image)

    # Get class probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities).item()

    # Load class names for ImageNet
    # You may need to replace or adapt this depending on the specific classes your model was trained on
    class_names = ["cat", "dog"]  # Replace with your actual class names
    ui.Q5_predict.setText("predict = " + class_names[predicted_class])

    probs = [prob.item() for prob in probabilities]

    # Create a bar chart
    plt.figure(figsize=(6, 6))
    plt.bar(class_names, probs, alpha=0.7)

    # Set chart title and axis labels
    plt.title("Probability of each class")
    plt.xlabel("Class Name")
    plt.ylabel("Probability")

    # Display probability values on top of the bars
    for i, prob in enumerate(probs):
        plt.text(i, prob, f"{prob:.2f}", ha="center", va="bottom")

    # Show the bar chart
    plt.xticks(rotation=45)  # Make x-axis labels more readable
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    ui.pushBotton.clicked.connect(load_image)
    ui.Q1_1_button.clicked.connect(Q1_1)
    ui.Q1_2_button.clicked.connect(Q1_2)
    ui.Q2_button.clicked.connect(Q2_1)
    ui.Q3_1_button.clicked.connect(Q3_1)
    ui.Q3_2_button.clicked.connect(Q3_2)
    ui.Q4_1_button.clicked.connect(Q4_1)
    ui.Q4_2_button.clicked.connect(Q4_2)
    ui.Q4_3_button.clicked.connect(Q4_3)
    ui.Q4_4_button.clicked.connect(Q4_4)
    ui.Q5_load_button.clicked.connect(load_image5)
    ui.Q5_1_button.clicked.connect(Q5_1)
    ui.Q5_2_button.clicked.connect(Q5_2)
    ui.Q5_3_button.clicked.connect(Q5_3)
    ui.Q5_4_button.clicked.connect(Q5_4)
    scene = DrawingScene()
    ui.Q4_graphicview.setScene(scene)

    MainWindow.show()
    app.exec_()
