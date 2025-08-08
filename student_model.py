import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torchvision import datasets, transforms
from copy import deepcopy
from teacher_model import ResNet18Teacher
from torch.utils.data import DataLoader, Subset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparameters
batch_size = 64 
num_epochs = 40  
learning_rate = 1e-3
subset_size = 8000 

#  augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# student model
class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2Student, self).__init__()
        self.model = mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


# Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, student_logits, teacher_logits, targets):
        # Hard loss
        hard_loss = self.ce_loss(student_logits, targets)
        # Soft loss
        T = self.temperature
        soft_loss = self.kl_loss(
            nn.functional.log_softmax(student_logits / T, dim=1),
            nn.functional.softmax(teacher_logits / T, dim=1)
        ) * (T * T)
        # Combine
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# Training Student with Teacher
def train_student(teacher_model, train_loader, test_loader, device,
                  num_classes=10, num_epochs=10, lr=1e-3, temperature=4.0, alpha=0.5):
    
    # Teacher is frozen
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    
    # Init student
    student_model = MobileNetV2Student(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    best_acc = 0
    best_wts = None
    
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            
            loss = criterion(student_outputs, teacher_outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * correct / total
        test_acc = evaluate(student_model, test_loader, device)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = deepcopy(student_model.state_dict())
            print(f"New best student model: {best_acc:.2f}%")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Save best model
    torch.save({
        'model_state_dict': best_wts,
        'best_acc': best_acc
    }, 'best_student.pth')
    
    return student_model


# Evaluation
def evaluate(model, loader, device):
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


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    #  dataset
    print("Loading datasets...")
    full_train_data = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=False)
    test_data = datasets.CIFAR10(root="./data", train=False, transform=transform_test)

    np.random.seed(42)
    subset_indices = np.random.choice(len(full_train_data), subset_size, replace=False)
    train_data = Subset(full_train_data, subset_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    teacher_model = ResNet18Teacher().to(device)
    teacher_model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    
    train_student(teacher_model, train_loader, test_loader, device,
                  num_classes=10, num_epochs=10, lr=1e-3, temperature=4.0, alpha=0.5)
