import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

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

vgg19_bn.load_state_dict(torch.load("vgg19_final.pt", map_location=device))  # 請確保路徑正確
vgg19 = vgg19_bn.to(device)

# 設置模型為評估模式
vgg19.eval()

# 載入圖片並進行預處理
transform = Compose([Resize((32, 32)), ToTensor(), Normalize(mean, std)])

image = Image.open("1.png").convert("L")
# image = cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)
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

# 輸出結果
print("Predicted class: {} ({})".format(class_names[predicted_class], predicted_class))
print("Class probabilities:")
for i, prob in enumerate(probabilities):
    print("{}: {:.2f}%".format(class_names[i], prob * 100))

# ui.predict_label.setText("Predicted = " + class_names[predicted_class])

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
# plt.show()
# plt.savefig("12.jpg")
