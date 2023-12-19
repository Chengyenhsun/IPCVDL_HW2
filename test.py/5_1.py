import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


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
