import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Hyperparameter
BATCH_SIZE = 100  # 根据你的需求调整
NUM_CLASSES = 2
# Loss function
loss_func = nn.CrossEntropyLoss()
# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 数据集路径
data_folder = './dataset'
train_dataset_path = os.path.join(data_folder, 'training_dataset')
val_dataset_path = os.path.join(data_folder, 'validation_dataset')

# 多进程需要加一个main函数，否则会报错
if __name__ == "__main__":
    # 数据增强-->训练集
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])
    train_set = torchvision.datasets.ImageFolder(train_dataset_path, transform=train_transform)
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

    # -->验证集
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    val_set = torchvision.datasets.ImageFolder(val_dataset_path, transform=val_transform)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=6)


# 定义训练的辅助函数 包含error与accuracy
def eval(model, loss_func, dataloader):
    model.eval()
    loss, accuracy = 0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = loss_func(logits, batch_y)
            loss += error.item()

            _, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y == batch_y.data).float().sum() / batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy * 100.0 / len(dataloader)
    return loss, accuracy


def train_epoch(model, loss_func, optimizer, dataloader):
    model.train()
    for batch_x, batch_y in tqdm(dataloader, desc="Training", ncols=100):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        error = loss_func(logits, batch_y)
        error.backward()
        optimizer.step()


# Load the pre-trained ResNet-50 model
resnet50 = models.resnet50(weights=None)
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, NUM_CLASSES)

# # Check the modified ResNet-50 model
# print(resnet50)

nepochs = 30
resnet50 = resnet50.to(device)
optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

print("Start training")


for epoch in range(nepochs):
    since = time.time()
    train_epoch(resnet50, loss_func, optimizer, train_dl)

    if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次模型
        checkpoint_filename = f"2resnet50_checkpoint_epoch_{epoch+1}.pt"
        torch.save(resnet50.state_dict(), checkpoint_filename)

    tr_loss, tr_acc = eval(resnet50, loss_func, train_dl)
    val_loss, val_acc = eval(resnet50, loss_func, val_dl)

    train_loss_list.append(tr_loss)
    val_loss_list.append(val_loss)

    train_acc_list.append(tr_acc.item())
    val_acc_list.append(val_acc.item())
    now = time.time()
    print(
        "[%2d/%d, %.0f seconds]|\t Train Loss: %.1e, Train Acc: %.2f\t |\t Val Loss: %.1e, Val Acc: %.2f"
        % (epoch + 1, nepochs, now - since, tr_loss, tr_acc, val_loss, val_acc)
    )

# 最後保存整個模型（非checkpoint）
torch.save(resnet50.state_dict(), "2resnet50_final.pt")

plt.subplot(2, 1, 1)
plt.plot(range(1, nepochs + 1), train_loss_list, label="Train_loss")
plt.plot(range(1, nepochs + 1), val_loss_list, label="Val_loss")
plt.legend(loc="upper right")
plt.ylabel("Loss")
plt.xlabel("Epoch")

plt.subplot(2, 1, 2)
plt.plot(range(1, nepochs + 1), train_acc_list, label="Train_acc")
plt.plot(range(1, nepochs + 1), val_acc_list, label="Val_acc")
plt.legend(loc="upper right")
plt.ylabel("Accuracy(%)")
plt.xlabel("Epoch")
plt.subplots_adjust(hspace=0.5)

# plt.savefig("learning_history.jpg")
# plt.show()
plt.savefig("learning_history.jpg")
