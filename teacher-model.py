import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from copy import deepcopy
import numpy as np
import time
from multiprocessing import Pool

# configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparameters
batch_size = 64 
num_epochs = 10  
learning_rate = 0.1
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

#  dataset
print("Loading datasets...")
full_train_data = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
test_data = datasets.CIFAR10(root="./data", train=False, transform=transform_test)

# random subset
np.random.seed(42)  # For reproducibility
subset_indices = np.random.choice(len(full_train_data), subset_size, replace=False)
train_data = Subset(full_train_data, subset_indices)

# dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# Model defiition
class ResNet50_model(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50_model, self).__init__()
        self.model = resnet18(pretrained=True)
        
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

print("Initializing model...")
model = ResNet50_model().to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs)

#  best model
best_acc = 0
best_model_wts = None

# Test function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_acc = 100.0 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc

# Training function
def train():
    global best_acc, best_model_wts
    
    print(f"Training on {subset_size} images for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # back pass
            loss.backward()
            optimizer.step()
            
            # statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
        
        # change learning rate
        scheduler.step()
        
        #  epoch statistics
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # test current model
        current_acc = test()
        
        # save best model
        if current_acc > best_acc:
            best_acc = current_acc
            best_model_wts = deepcopy(model.state_dict())
            print(f"New best model! Accuracy: {best_acc:.2f}%")
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Complete | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    
    # save best model
    torch.save({
        'model_state_dict': best_model_wts,
        'best_acc': best_acc,
        'epochs': num_epochs,
    }, 'best_model.pth')
    print(f"Saved best model with accuracy: {best_acc:.2f}%")
    

#  training
# if __name__ == '__main__':
#     import multiprocessing
#     multiprocessing.freeze_support()
#     # train()
#     test()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    model.eval()
    
    test()