import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import time
import torchvision.models as models
from tqdm import tqdm

# Hyperparameter
BATCH_SIZE = 64  # Reducing batch size for demonstration purposes, adjust as needed
# Loss function
loss_func = (
    nn.BCEWithLogitsLoss()
)  # Binary Cross Entropy loss for binary classification
# check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.data.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], torch.tensor(
            self.data[idx][1]
        )  # Remove the extra dimension


def eval(model, loss_func, dataloader):
    model.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = loss_func(
                logits.squeeze(), batch_y.float()
            )  # Squeezing logits for BCEWithLogitsLoss
            loss += error.item()

            predicted = torch.round(torch.sigmoid(logits))
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    loss /= len(dataloader)
    accuracy = correct / total * 100.0
    return loss, accuracy


def train_epoch(model, loss_func, optimizer, dataloader):
    model.train()
    for batch_x, batch_y in tqdm(dataloader, desc="Training", ncols=100):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        error = loss_func(
            logits.squeeze(), batch_y.float()
        )  # Squeezing logits for BCEWithLogitsLoss
        error.backward()
        optimizer.step()


if __name__ == "__main__":
    training_path = "dataset/training_dataset"
    training_dataset = CustomDataset(
        root_dir=training_path,
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(),
            ]
        ),
    )
    train_dl = DataLoader(
        training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    validation_path = "dataset/validation_dataset"
    validation_dataset = CustomDataset(
        root_dir=validation_path,
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        ),
    )
    val_dl = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=0)

    RN_random = models.resnet50(pretrained=True)  # Using pretrained weights
    num_ftrs = RN_random.fc.in_features
    # Replace the output layer with a FC layer of 1 node (for binary classification)
    RN_random.fc = nn.Sequential(nn.Linear(num_ftrs, 1))

    nepochs = 5  # Increased the number of epochs for better training
    RN_re = RN_random.to(device)

    optimizer = torch.optim.SGD(
        RN_re.parameters(), lr=0.01, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40], gamma=0.1
    )

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(nepochs):
        since = time.time()
        train_epoch(RN_re, loss_func, optimizer, train_dl)

        tr_loss, tr_acc = eval(RN_re, loss_func, train_dl)
        val_loss, val_acc = eval(RN_re, loss_func, val_dl)

        train_loss_list.append(tr_loss)
        val_loss_list.append(val_loss)

        train_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)
        now = time.time()
        print(
            "[%2d/%d, %.0f seconds]|\t Train Loss: %.4f, Train Acc: %.2f%%\t |\t Val Loss: %.4f, Val Acc: %.2f%%"
            % (epoch + 1, nepochs, now - since, tr_loss, tr_acc, val_loss, val_acc)
        )

    torch.save(RN_re.state_dict(), "RN_re_final.pt")
