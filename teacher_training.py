import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from utils import accuracy, validate, save_checkpoint

def main():
# preprocessing
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    # Download CIFAR-100 dataset (Teacher training)
    train_data = datasets.CIFAR100(root="./data/cifar100", train=True, download=True, transform=transform_train)
    val_data = datasets.CIFAR100(root="./data/cifar100", train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet 50 
    teacher = timm.create_model("resnet50", pretrained=True, num_classes=100)  # CIFAR-100 has 100 classes
    teacher = teacher.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(teacher.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Training Config
    epochs = 50
    patience = 5  
    best_val_acc = 0.0
    early_stop_counter = 0

# training loop
    for epoch in range(epochs):
        teacher.train()
        running_loss = 0.0
        running_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = teacher(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels) * images.size(0)

        scheduler.step()

        avg_loss = running_loss / len(train_loader.dataset)
        avg_train_acc = running_acc / len(train_loader.dataset)

# validation
        val_acc = validate(teacher, val_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

# save checkpoints
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(teacher, "resnet50_teacher_best.pth")
            print(f"best model saved with Val Accuracy {val_acc*100:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs")

        if early_stop_counter >= patience:
            print(f"Early stopped Best Val Accuracy: {best_val_acc*100:.2f}%")
            break

    print("Teacher training complete.")

if __name__ == "__main__":
    main()
