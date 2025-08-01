import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from utils import accuracy, validate, save_checkpoint

# Knowledge Distillation Loss (KL + CE)
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss (student vs true labels)
        hard_loss = self.ce(student_logits, labels)
        # Soft loss (student vs teacher predictions)
        soft_loss = self.kl(
            nn.functional.log_softmax(student_logits / self.temperature, dim=1),
            nn.functional.softmax(teacher_logits / self.temperature, dim=1),
        ) * (self.temperature ** 2)
        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss

def main():
# preprocessing 
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])

    train_data = datasets.CIFAR10(root="./data/cifar10", train=True, download=True, transform=transform_train)
    val_data = datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet-50 Teacher 
    teacher = timm.create_model("resnet50", pretrained=False, num_classes=100)
    teacher.load_state_dict(torch.load("resnet50_teacher_best.pth", map_location=device))
    teacher = teacher.to(device)
    teacher.eval()  # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

# MobileNet Student 
    student = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=10)
    student = student.to(device)

    criterion = DistillationLoss(temperature=4.0, alpha=0.7)
    optimizer = optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


    epochs = 50
    patience = 5  
    best_val_acc = 0.0
    early_stop_counter = 0

# training loop 
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        running_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(images)

            optimizer.zero_grad()
            student_outputs = student(images)

            # KD Loss
            loss = criterion(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(student_outputs, labels) * images.size(0)

        scheduler.step()

        avg_loss = running_loss / len(train_loader.dataset)
        avg_train_acc = running_acc / len(train_loader.dataset)

# validation
        val_acc = validate(student, val_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

# save checkpoints
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(student, "mobilenet_student_best.pth")
            print(f"Best Student model saved with Val Accuracy {val_acc*100:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs")

        if early_stop_counter >= patience:
            print(f"Early stopped. Best Val Accuracy: {best_val_acc*100:.2f}%")
            break

    print("Student training with Knowledge Distillation complete.")

if __name__ == "__main__":
    main()
