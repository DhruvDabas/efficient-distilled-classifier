import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from copy import deepcopy
import numpy as np

# configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[teacher_model] Using device: {device}")

# hyperparameters (tweak as needed)
batch_size = 64
num_epochs = 10
learning_rate = 1e-3   # actual optimizer lr used below
subset_size = 8000
num_workers = min(8, os.cpu_count() or 0)
pin_memory = True if torch.cuda.is_available() else False

# transforms (resize to 224 for ResNet)
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


class ResNet18_model(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ResNet18 teacher model
class ResNet18Teacher(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Teacher, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def load_dataloaders(subset_size=subset_size):
    """Return train_loader, test_loader. subset_size can be None or <=0 to use full dataset."""
    full_train = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)

    # guard subset_size
    if subset_size is None or subset_size <= 0 or subset_size >= len(full_train):
        train_dataset = full_train
        print(f"[teacher_model] Using full train dataset: {len(full_train)} images.")
    else:
        np.random.seed(42)
        subset_indices = np.random.choice(len(full_train), subset_size, replace=False)
        train_dataset = Subset(full_train, subset_indices)
        print(f"[teacher_model] Using subset of train dataset: {subset_size} images.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def train_loop(model, train_loader, test_loader, num_epochs=num_epochs, lr=learning_rate):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # OneCycleLR should be stepped every batch: set total_steps = epochs * steps_per_epoch
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs
    )

    best_acc = 0.0
    best_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()   # step per batch

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = deepcopy(model.state_dict())
            print(f"[teacher_model] New best model: {best_acc:.2f}%")

        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} Train Acc: {train_acc:.2f}% Test Acc: {test_acc:.2f}% "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save best weights (if found)
    if best_wts is not None:
        torch.save({'model_state_dict': best_wts, 'best_acc': best_acc, 'epochs': num_epochs}, 'best_model.pth')
        print(f"[teacher_model] Saved best_model.pth with acc {best_acc:.2f}%")
    else:
        print("[teacher_model] No best model to save.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    train_loader, test_loader = load_dataloaders(subset_size=subset_size)

    model = ResNet18Teacher(num_classes=10, pretrained=True).to(device)
    print("[teacher_model] Model initialized.")

    # If checkpoint exists, optionally load it for evaluation/test, else train
    ckpt_path = 'best_model.pth'
    if os.path.exists(ckpt_path):
        print("[teacher_model] Found checkpoint; loading for evaluation.")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        acc = evaluate(model, test_loader)
        print(f"[teacher_model] Loaded checkpoint test accuracy: {acc:.2f}%")
    else:
        print("[teacher_model] No checkpoint found — starting training.")
        train_loop(model, train_loader, test_loader)
