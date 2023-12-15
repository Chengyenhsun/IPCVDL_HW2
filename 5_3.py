import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from tqdm import tqdm

# 設定一些參數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32  # 請根據需要調整批次大小

# 載入模型
with_RE = 'resnet50/with_RE/2resnet50_checkpoint_epoch_20.pt'
without_RE = 'resnet50/without_RE/resnet50_checkpoint_epoch_20.pt'
model_paths = [without_RE, with_RE]
models = []

for path in model_paths:
    model = resnet50(pretrained=False, num_classes=2)  # 請確保 num_classes 是你的問題中需要的類別數量
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    models.append(model)

# 載入驗證資料集
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

validation_dataset = datasets.ImageFolder(root='dataset/validation_dataset', transform=transform)  # 請替換為你的實際路徑

# 計算準確度
accuracies = []
with torch.no_grad():
    for model in models:
        correct = 0
        total = 0
        dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100  # Convert accuracy to percentage
        accuracies.append(accuracy)

# 顯示長條圖
fig, ax = plt.subplots()
bars = ax.bar(['Without Random-Erasing', 'With Random-Erasing'], accuracies)

# Add numerical values on top of the bars
for bar, accuracy in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{accuracy:.2f}%', ha='center', va='bottom', color='black', fontweight='bold')

plt.xlabel('Models')
plt.ylabel('Accuracy(%)')
plt.title('Accuracy Comparison')
plt.savefig("Accuracy_Comparison.jpg")
plt.show()
