import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm

print(torch.backends.mps.is_available())  # the MacOS is higher than 12.3+
print(torch.backends.mps.is_built())  # MPS is activated

# 超参数
BATCH_SIZE = 100
# 损失函数
loss_func = nn.CrossEntropyLoss()
# 可以在CPU或者GPU上运行
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# MNIST的输入图片只有一个channel，所以修改均值和标准差
mean = [0.5]
std = [0.5]
n_train_samples = 60000

# 多进程需要加一个main函数，否则会报错
if __name__ == "__main__":
    # 数据增强-->训练集
    train_set = dsets.MNIST(
        root="./MNIST/",
        train=True,
        download=True,
        transform=trans.Compose(
            [
                trans.RandomRotation(10),  # Rotate randomly by 10 degrees
                trans.ToTensor(),
                trans.Normalize(mean, std),
            ]
        ),
    )
    train_dl = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )  # 如需多线程，可以自行更改

    #  -->测试集
    val_set = dsets.MNIST(
        root="./MNIST/",
        train=False,
        download=True,
        transform=trans.Compose([trans.ToTensor(), trans.Normalize(mean, std)]),
    )

    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)  # 如需多线程，可以自行更改


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

            probs, pred_y = logits.data.max(dim=1)
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


# Load the pre-trained VGG19 model with Batch Normalization
vgg19_bn = models.vgg19_bn(pretrained=False)

# Modify the first layer to accept a single channel input
vgg19_bn.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

# Remove the last max pooling layer to better suit the smaller input size
vgg19_bn.features = vgg19_bn.features[:-1]

# Check the modified VGG19 model
print(vgg19_bn)

nepochs = 30
vgg19 = vgg19_bn.to(device)
optimizer = torch.optim.SGD(vgg19.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

print("Start training VGG19 on MNIST")

for epoch in range(nepochs):
    since = time.time()
    train_epoch(vgg19, loss_func, optimizer, train_dl)

    if (epoch + 1) % 5 == 0:  # 每5個epoch保存一次模型
        checkpoint_filename = f"vgg19_checkpoint_epoch_{epoch+1}.pt"
        torch.save(vgg19.state_dict(), checkpoint_filename)

    tr_loss, tr_acc = eval(vgg19, loss_func, train_dl)
    val_loss, val_acc = eval(vgg19, loss_func, val_dl)

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
torch.save(vgg19.state_dict(), "vgg19_final.pt")

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
plt.show()
plt.savefig("learning_history.jpg")
